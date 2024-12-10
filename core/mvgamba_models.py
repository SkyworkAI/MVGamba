import math

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd, autocast
import pdb

from .gambaformer import GambaFormer

from core.typings import *
from core.options import Options
from core.gs import GaussianRenderer
from kiui.lpips import LPIPS

import random

from torch.utils.checkpoint import checkpoint


inverse_sigmoid = lambda x: np.log(x / (1 - x))
custom_normalize = lambda x: F.normalize(x.float(), dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        # self.output_activation = self.make_activation(output_activation)
        self.output_activation = self.make_activation(activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

class RotationNet(nn.Module):
    canonical_quaternions = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [ 0.7071,  0.7071,  0.0000,  0.0000],
        [ 0.7071,  0.0000,  0.7071,  0.0000],
        [ 0.7071,  0.0000,  0.0000,  0.7071],
        [ 0.0000,  0.7071,  0.7071,  0.0000],
        [ 0.0000,  0.7071,  0.0000,  0.7071],
        [ 0.0000,  0.0000,  0.7071,  0.7071],
        [ 0.7071, -0.7071,  0.0000,  0.0000],
        [ 0.7071,  0.0000, -0.7071,  0.0000],
        [ 0.7071,  0.0000,  0.0000, -0.7071],
        [ 0.0000,  0.7071, -0.7071,  0.0000],
        [ 0.0000,  0.7071,  0.0000, -0.7071],
        [ 0.0000,  0.0000,  0.7071, -0.7071],
        [-0.7071,  0.7071,  0.0000,  0.0000],
        [-0.7071,  0.0000,  0.7071,  0.0000],
        [-0.7071,  0.0000,  0.0000,  0.7071],
        [ 0.0000, -0.7071,  0.7071,  0.0000],
        [ 0.0000, -0.7071,  0.0000,  0.7071],
        [ 0.0000,  0.0000, -0.7071,  0.7071],
        [-0.7071, -0.7071,  0.0000,  0.0000],
        [-0.7071,  0.0000, -0.7071,  0.0000],
        [-0.7071,  0.0000,  0.0000, -0.7071],
        [ 0.0000, -0.7071, -0.7071,  0.0000],
        [ 0.0000, -0.7071,  0.0000, -0.7071],
        [ 0.0000,  0.0000, -0.7071, -0.7071]
    ]).float()

    @staticmethod
    def forward(x):
        prob = F.softmax(x.float(), dim=-1)
        quaternion = torch.sum(prob.unsqueeze(-1) * RotationNet.canonical_quaternions.to(x.device), dim=-2)
        rot = custom_normalize(quaternion)
        return rot

class RotationNet_Hard(nn.Module):
    def __init__(self, opt):
        super(RotationNet_Hard, self).__init__()
        self.register_buffer("canonical_quaternions", RotationNet.canonical_quaternions)
        
        self.temperature = opt.temperature
        self.max_temp = opt.temperature
        self.min_temp = opt.min_temp
        self.temperature_decay = opt.temperature_decay
        self.straight_through = opt.straight_through

    def set_num_updates(self, num_updates):
        self.temperature = max(
            self.max_temp * self.temperature_decay**num_updates, self.min_temp
        )
    def update_temperature(self):
        self.temperature = max(
            self.temperature * self.temperature_decay, self.min_temp
        )

    def forward(self, x):
        if self.training:
            prob = F.gumbel_softmax(x.float(), tau=self.temperature, hard=self.straight_through)
            assert prob.dtype == torch.float32
            self.update_temperature()
        else:
            idx = x.argmax(dim=-1)
            prob = F.one_hot(idx, num_classes=x.size(-1)).float()
        quaternion = torch.sum(prob.unsqueeze(-1) * self.canonical_quaternions.to(x.device), dim=-2)
        rot = custom_normalize(quaternion)
        return rot

class GSDecoder(nn.Module):
    def __init__(self, transformer_dim, 
                 SH_degree,
                 opt,
                 init_density=0.1, 
                 clip_scaling=0.1):
        super(GSDecoder, self).__init__()
        
        self.mlp_dim = transformer_dim
        self.clip_scaling = clip_scaling

        self.mlp_net = MLP(transformer_dim, self.mlp_dim, n_neurons=transformer_dim * 4, n_hidden_layers=1, activation="silu")

        pos_bound = 0.8
        coords = torch.linspace(-1 * pos_bound, pos_bound, 21)[None, :].repeat(3, 1)
        self.register_buffer("coords", coords)
        # xyz (3) + scale (3) + rot(4) + opacity(1) + SH_0 (3)
        self.pred_keys = ["xyz", "opacity", "scale", "rot", "rgb"]

        self.fix_keys = []

        self.gs_layer = nn.ModuleDict()
        for key in self.pred_keys:
            if key in self.fix_keys:
                continue
            if key == "xyz":
                layer = nn.Linear(self.mlp_dim, 3 * self.coords.size(-1))
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias,0)
            elif key == "scale":
                layer = nn.Linear(self.mlp_dim, 3)
                nn.init.constant_(layer.bias, -1.8)
            elif key == "rot":
                layer = nn.Linear(self.mlp_dim, 32)
                nn.init.constant_(layer.bias, 0)
                if opt.use_gumbel_softmax:
                    self.get_rot = RotationNet_Hard(opt)
                else:
                    self.get_rot = RotationNet()
            elif key == "opacity":
                layer = nn.Linear(self.mlp_dim, 1)
                nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            elif key == "shs":
                shs_dim = 3 * (SH_degree + 1) ** 2
                layer = nn.Linear(self.mlp_dim, int(shs_dim))
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            elif key == "rgb":
                color_dim = 3
                layer = nn.Linear(self.mlp_dim, color_dim)
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0.0)
            else:
                raise NotImplementedError
            self.gs_layer[key] = layer
    
    def forward(self, feats):  # (bsz, num_pts, feat_dim)
        gsparams = []
        feats = self.mlp_net(feats)
        for key in self.pred_keys:
            if key in self.fix_keys:
                if key == "scale":
                    fix_v = 0.03 * torch.ones(*feats.shape[:2], 3).to(feats.device)
                elif key == "rot":
                    fix_v = torch.zeros(*feats.shape[:2], 4).to(feats.device)
                    fix_v[:, :, 0] = 1.
                gsparams.append(fix_v)
                continue
            v = self.gs_layer[key](feats)
            if key == "xyz":
                # (bsz, num_pts, 3, prob_num)
                v = v.reshape(*v.shape[:2], 3, -1) 
                prob = F.softmax(v, dim=-1)
                assert prob.dtype == torch.float32
                # coords shape (1, 1, 3, prob_num)
                v = (prob * self.coords[None, None]).sum(dim=-1)
            elif key == "scale":
                v = 0.1 * F.softplus(v)
            elif key == "rot":
                v = self.get_rot(v)
            elif key == "opacity":
                v = torch.sigmoid(v)
            elif key == "shs":
                pass 
            elif key == "rgb":
                v = torch.sigmoid(v)
            else:
                raise NotImplementedError
            gsparams.append(v)

        return torch.cat(gsparams, dim=-1)


# The most basic version with multi-head attention
class GSPredictor(nn.Module):
    def __init__(self, 
                 opt: Options,
                 SH_degree=0,
                 **kwargs):
        super().__init__()
        
        if opt.plucker_ray:
            input_channels = 9
        else:
            input_channels = 3
        self.map_image = nn.Conv2d(
            in_channels=input_channels, out_channels=opt.gamba_dim, kernel_size=opt.patch_size, stride=opt.patch_size,
        )
        if isinstance(self.map_image, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.map_image.in_channels * self.map_image.kernel_size[0] * self.map_image.kernel_size[1]
            nn.init.trunc_normal_(self.map_image.weight, std=math.sqrt(1 / fan_in))
            if self.map_image.bias is not None:
                nn.init.zeros_(self.map_image.bias)
        
        if opt.upsampler_type == 'conv2d':
            self.upsampler = nn.ConvTranspose2d(opt.gamba_dim, opt.upsampler_dim, kernel_size=opt.upsampler_kernel, stride=opt.upsampler_kernel, padding=0)
            out_channels = opt.upsampler_dim
            gs_num = ((opt.input_size // opt.patch_size) ** 2) * 2 * opt.num_input_views
        elif opt.upsampler_type == 'conv1d':
            self.upsampler = nn.ConvTranspose1d(opt.gamba_dim, opt.upsampler_dim, kernel_size=opt.upsampler_kernel, stride=opt.upsampler_kernel, padding=0)
            out_channels = opt.upsampler_dim
            gs_num = ((opt.input_size // opt.patch_size) ** 2) * 2 * opt.num_input_views
        else:
            out_channels = opt.gamba_dim
            gs_num = ((opt.input_size // opt.patch_size) ** 2) * 4 * opt.num_input_views
        
        # default init
        self.initialize_weights()
                            
        # custom init      
        self.transformer = GambaFormer(inner_dim=opt.gamba_dim, 
                                         num_layers=opt.gamba_layers, 
                                         gs_num=gs_num, drop_path_rate=0.1,
                                         rms_norm=opt.rms_norm, fused_add_norm=opt.fused_add_norm, residual_in_fp32=opt.residual_in_fp32,
                                         grad_checkpointing=opt.grad_checkpointing, use_pos_embed=opt.use_pos_embed)

        self.decoder = GSDecoder(out_channels, SH_degree, opt=opt)
    
        self.opt = opt

    
    def initialize_weights(self):
    # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.opt.input_size // self.opt.patch_size
        x = tokens
        if self.opt.upsampler_type == 'conv2d':
            x = x.view(N, 16, H, W, -1)
            x = torch.einsum('nihwd->indhw', x)  # [16, N, D, H, W]
            x = x.contiguous().view(16*N, -1, H, W)  # [16*N, D, H, W]
            x = self.upsampler(x)  # [16*N, D', H', W']
            x = x.view(16, N, *x.shape[-3:])  # [16, N, D', H', W']
            x = torch.einsum('indhw->nihwd', x)  # [N, 16, H', W', D']
            x = x.reshape(N, -1, x.shape[-1]) # [N, 16*H'*W', D']
            x = x.contiguous()
        elif self.opt.upsampler_type == 'conv1d':
            x = x.view(N, -1, self.opt.gamba_dim)
            x = torch.einsum('nlc->ncl', x).contiguous()  # [N, C, L]
            x = self.upsampler(x) # [N, C', L']
            x = torch.einsum('ncl->nlc', x).contiguous() # [N, L', C']
        return x
        
    def forward(self, cond_views):
        """
        Input: noise views and class label with source camera pose 
        Output: clean gaussian splatting parameters 
        """
        
        # gs [N,gs_num, gs_dim:14]
        bsz, cond_num, c, h, w = cond_views.size()
        cond_views = cond_views.view(bsz*cond_num, c, h, w)

        # (bsz * views, H * W, C)
        img_cond = self.map_image(cond_views) 
        # (bsz * views, C, H * W)
        img_cond_1 = img_cond.flatten(2)
        img_cond_2 = img_cond.permute(0, 1, 3, 2).flatten(2)
        img_cond_3 = img_cond.flip(dims=[3]).flatten(2)
        img_cond_4 = img_cond.permute(0, 1, 3, 2).flip(dims=[3]).flatten(2)
        # (bsz * view_num, C, token_num)
        if self.opt.upsampler_type == 'none':
            img_cond = torch.cat([img_cond_1, img_cond_2, img_cond_3, img_cond_4], dim=-1)
        else:
            img_cond = torch.cat([img_cond_1, img_cond_3], dim=-1)
        img_cond = img_cond.permute(0, 2, 1).reshape(bsz, -1, self.opt.gamba_dim)

        feats = self.transformer(img_cond)
        feats = self.reshape_upsample(feats)
        gs = self.decoder(feats)
        outputs = {"pred_gs": gs}
        return outputs


class MVGamba(torch.nn.Module):
    def __init__(self,
        opt: Options,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        """
        use opt to input params
        """
        super().__init__()
        self.model = GSPredictor(opt, **model_kwargs)
        self.opt = opt
        self.gs_render = GaussianRenderer(opt)
        if opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        
        self.dtype = torch.bfloat16 if opt.mixed_precision == 'bf16' else torch.float32
    
    def forward_gaussians(self, images, cam_poses=None):
        # images: [B, 4, 3, H, W]
        # return: Gaussians: [B, dim_t]

        # B, V, C, H, W = images.shape
        decoder_out = self.model(cond_views=images.type(self.dtype))
        
        return decoder_out['pred_gs']

    
    def prepare_default_rays(self, device, elevation=0):
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def forward(self, data, epoch, step_ratio = 1, vis = 0):
        results = {}
        loss = 0

        cond_poses = data['input']['camposes']
        cond_views = data['input']['images']

        #flatten
        cond_poses = cond_poses.view(cond_poses.size(0), cond_poses.size(1), -1)  # (bsz, view_num, 16)

        decoder_out = self.model(cond_views=cond_views.type(self.dtype), # (bsz, view_num, c, h, w)
                                 cam_poses=cond_poses.type(self.dtype)) # (bsz, view_num, 16)
        
        with autocast(enabled=False):
            # bg aug
            if not self.training or random.random() > self.opt.prob_bg_color:
                bg_color = torch.ones(3, dtype=torch.float32, device=cond_views.device)
            else:
                # random r, g, b
                r_color = torch.tensor([1, 0, 0], dtype=torch.float32, device=cond_views.device)
                g_color = torch.tensor([0, 1, 0], dtype=torch.float32, device=cond_views.device)
                b_color = torch.tensor([0, 0, 1], dtype=torch.float32, device=cond_views.device)
                bg_color = random.choice([r_color, g_color, b_color])
            # use the other views for rendering and supervision
            results = self.gs_render.render(decoder_out['pred_gs'], data['cam_view'], 
                                            data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)

        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['normals_pred'] = results["render_normal"]
        results['surf_pred'] = results["surf_normal"]

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        # print(f"gt_images: {gt_images.dtype}, pred_images: {pred_images.dtype}, bg_color: {bg_color.dtype}")
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0 and epoch >= self.opt.start_lpips :
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False).type(self.dtype), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False).type(self.dtype),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
        
        # regularization on rot, opacity
        if self.opt.lambda_reg > 0:
            loss_reg = 0.
            opacity = decoder_out['pred_gs'][:, :, 3:4].contiguous().float()
            loss_reg += (1. - opacity).mean()
            results['loss_reg'] = loss_reg
            loss = loss + self.opt.lambda_reg * loss_reg


        if self.opt.lambda_normal > 0 and epoch >= self.opt.start_normal :
            # [bsz, views, 3, H, W]
            normal_error = (1 - (results['normals_pred'] * results['surf_pred']).sum(dim=2))[:, :, None]
            loss_normal = self.opt.lambda_normal * (normal_error).mean()
            loss = loss + loss_normal
            
        if self.opt.lambda_dist > 0 and epoch >= self.opt.start_dist :
            loss_dist = self.opt.lambda_dist * (results['render_dist']).mean()
            loss = loss + loss_dist

        results['loss'] = loss

        assert loss_mse.dtype == torch.float32
        assert loss.dtype == torch.float32

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        
            if vis == 1: # scale = 0 for point visulization
                with autocast(enabled=False):
                    results_points = self.gs_render.render(decoder_out['pred_gs'], data['cam_view'], 
                                                data['cam_view_proj'], data['cam_pos'], bg_color=torch.ones(3, dtype=torch.float32, device=cond_views.device), scale_modifier=0.0)
                    pred_points = results_points['image'] # [B, V, C, output_size, output_size]
                    results['pred_points'] = pred_points
            
        return results
        
