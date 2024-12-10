import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    model_type: str = 'gamba'
    plucker_ray: bool = True

    input_size: int = 448
    num_input_views: int = 4 # set input view as 4

    # patch
    patch_size: int = 14
    
    # model params
    gs_num: int = 0
    gamba_layers: int = 18
    gamba_dim: int = 768
    upsampler_type: str = 'none' # ['conv2d', 'conv1d', 'none']

    # GambeFormer
    rms_norm: bool = True
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    grad_checkpointing: bool = True
    use_pos_embed: bool = True

    # gs decoder
    use_gumbel_softmax: bool = False
    temperature: float = 2.0
    min_temp: float = 0.5
    temperature_decay: float = 0.999995
    straight_through: bool = True


    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!    
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)

    splat_size: int = 64
    # gaussian render size
    output_size: int = 512 # output size

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 10
    # number of views
    # num_input_views : int = 1
    num_output_views: int = 6
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### training
    # workspace
    workspace: str = 'workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 900
    # lpips loss weight
    lambda_lpips: float = 0.6
    start_lpips: int = 0
    # reg loss weight
    lambda_reg: float = 0.001
    lambda_normal: float = 0.05
    lambda_dist : float = 0.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 1e-3
    # weight decay
    weight_decay: float = 0.05
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    # augmentation prob for background color
    prob_bg_color: float = 0.5
    # warmup iters for lr
    warmup_epochs: int = 20
    
    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    # test_order: int = 1
    
    # renderig resolution zoom factor for patched rendering
    zoom: int = 3
# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['mvgamba'] = 'the default settings for MVGamba'
config_defaults['mvgamba'] = Options()

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
