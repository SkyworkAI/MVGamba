import tyro
import time
import random
import datetime
import torch
from core.options import AllConfigs
from core.mvgamba_models import MVGamba
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs, set_seed
from safetensors.torch import load_file
from core.utils import CosineWarmupScheduler
import psutil

import os
import kiui
from torch.utils.tensorboard import SummaryWriter
import wandb


def needs_decay(param_name):
    if "pos_embed" in param_name:
        return False
    return True

def main():    
    set_seed(42)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    # model
    model = MVGamba(opt)

    # data
    if opt.data_mode == 's3':
        from core.provider_ikun import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError
    
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer, position embedding doesn/t need weight decay
    parameters = [{
        "params": [param for name, param in model.named_parameters() if needs_decay(name)],
        "weight_decay": opt.weight_decay,
    }, {
        "params": [param for name, param in model.named_parameters() if not needs_decay(name)],
        "weight_decay": 0.0,
    }]

    optimizer = torch.optim.AdamW(parameters, lr=opt.lr, betas=(0.9, 0.95))
    if accelerator.is_main_process:
        print(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    steps_per_epoch = len(train_dataloader) // opt.gradient_accumulation_steps
    total_steps = opt.num_epochs * steps_per_epoch
    warmup_iters = opt.warmup_epochs * steps_per_epoch 
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_iters=warmup_iters, max_iters=total_steps, min_lr=0.1*opt.lr) 

    # resume
    resume_after_prepare = False
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
            model.load_state_dict(ckpt['model'])
        elif opt.resume.endswith('.pth'):
            ckpt = torch.load(opt.resume, map_location='cpu')
            print(f"resume from {opt.resume}, loading ...")
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            print("load checkpoint done!")
        else:
            resume_after_prepare = True
        
    else:
        start_epoch = 0    

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    if resume_after_prepare:
        accelerator.load_state(opt.resume, strict=False)
        start_epoch = int(scheduler.scheduler._step_count // steps_per_epoch) + 1
        print(f"Resuming from {opt.resume} at epoch {start_epoch}")   
    if opt.use_gumbel_softmax:
        num_updates = max(0, start_epoch * len(train_dataloader))
        print(f"set gumbel_softmax num_updates: {num_updates}")
        model.module.model.decoder.get_rot.set_num_updates(num_updates)
        
    start_time = datetime.datetime.now()
    # loop
    if accelerator.is_main_process:
        writer = SummaryWriter(opt.workspace)
        wandb.init(
            project="nips_stable",
            config=opt,
            dir=opt.workspace,
            name=opt.workspace.split('/')[-1],
        )
        wandb.watch(model, log_freq=1000)
    
    for epoch in range(start_epoch, opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        total_loss_lpips = 0
        total_loss_reg = 0
        wandb_gt_image = None
        wandb_pred_image = None
        wandb_eval_gt_image = None
        wandb_eval_pred_image = None
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()
                
                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                
                out = model(data, epoch, step_ratio)
                
                loss = out['loss']
                psnr = out['psnr']

                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
                if 'loss_lpips' in out:
                    total_loss_lpips += out['loss_lpips'].detach()
                if 'loss_reg' in out:
                    total_loss_reg += out['loss_reg'].detach()

            if accelerator.is_main_process:
                # logging,changed depends on the category
                if i % 100 == 0 :
                    mem_free, mem_total = torch.cuda.mem_get_info()  
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                    elapsed = datetime.datetime.now() - start_time
                    elapsed_str = str(elapsed).split('.')[0]  
                    process = psutil.Process()
                    print(f"[{current_time} INFO] {i}/{len(train_dataloader)} | "
                        f"Elapsed: {elapsed_str} | "
                        f"Mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G | "
                        f"LR: {scheduler.get_last_lr()[0]:.7f} | "
                        f"Step ratio: {step_ratio:.4f} | "
                        f"Loss: {loss.item():.6f} | "
                        f"Memory per process: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
                
                # save log images
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    pred_normals = out["normals_pred"].to(dtype=torch.float32).detach().cpu().numpy() * 0.5 + 0.5
                    pred_normals = pred_normals.transpose(0, 3, 1, 4, 2).reshape(-1, pred_normals.shape[1] * pred_normals.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_normals_{epoch}_{i}.jpg', pred_normals)

                    # wandb log image
                    wandb_gt_image = wandb.Image(gt_images[::4, ::4, :], caption=f"train_gt_images")
                    wandb_pred_image = wandb.Image(pred_images[::4, ::4, :], caption=f"train_pred_images")
                    wandb_pred_normal = wandb.Image(pred_normals, caption=f"train_pred_normals_{epoch}_{i}")

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_loss_lpips = accelerator.gather_for_metrics(total_loss_lpips).mean()
        total_loss_reg = accelerator.gather_for_metrics(total_loss_reg).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_loss_lpips /= len(train_dataloader)
            total_loss_reg /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            writer.add_scalar("Loss/train", total_loss, epoch)
            writer.add_scalar("PSNR/train", total_psnr, epoch)
            wandb.log({"Loss/train": total_loss, "PSNR/train": total_psnr, 
                       "Loss/loss_lpips": total_loss_lpips, "Loss/loss_reg": total_loss_reg,
                       "LR/lr": scheduler.get_last_lr()[0]
                      }, step=epoch, commit=False)
            if opt.use_gumbel_softmax:
                wandb.log({"LR/temperature": model.module.model.decoder.get_rot.temperature}, step=epoch, commit=False)
            wandb.log({"train/gt_images": wandb_gt_image, 
                       "train/pred_images": wandb_pred_image,
                       "train/pred_normals": wandb_pred_normal,}, step=epoch, commit=False)
            # save psnr file
            train_psnr_log_file = os.path.join(opt.workspace, "train_psnr_log.txt")
            with open(train_psnr_log_file, "a") as file:
                file.write(f"Epoch: {epoch}, PSNR: {total_psnr.item():.4f}\n")
            
        
        # checkpoint
        if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=opt.workspace)
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.optimizer.state_dict(),
                    'scheduler': scheduler.scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(opt.workspace, 'checkpoint_ep{:03d}.pth'.format(epoch)))
            accelerator.wait_for_everyone()
            # torch.distributed.barrier()

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data, epoch, vis = 1)

                psnr = out['psnr']
                total_psnr += psnr.detach()

                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    pred_points = out['pred_points'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_points = pred_points.transpose(0, 3, 1, 4, 2).reshape(-1, pred_points.shape[1] * pred_points.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_points_{epoch}_{i}.jpg', pred_points)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

                    wandb_eval_gt_image = wandb.Image(gt_images[::4, ::4, :], caption=f"eval_gt_images")
                    wandb_eval_pred_image = wandb.Image(pred_images[::4, ::4, :], caption=f"eval_pred_images")

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                writer.add_scalar("PSNR/eval", total_psnr, epoch)
                # wandb.log({"PSNR/eval": total_psnr}, step=(epoch+1)*len(train_dataloader))
                wandb.log({"PSNR/eval": total_psnr}, step=epoch, commit=False)
                wandb.log({"eval/gt_images": wandb_eval_gt_image, "eval/pred_images": wandb_eval_pred_image}, step=epoch, commit=True)
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
                # save psnr file
                test_psnr_log_file = os.path.join(opt.workspace, "test_psnr_log.txt")
                with open(test_psnr_log_file, "a") as file:
                    file.write(f"Epoch: {epoch}, PSNR: {total_psnr.item():.4f}\n")
            



if __name__ == "__main__":
    main()
