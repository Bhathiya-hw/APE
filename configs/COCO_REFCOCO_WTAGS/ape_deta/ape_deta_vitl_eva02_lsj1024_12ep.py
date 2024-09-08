from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.config import get_config
from ape.modeling.backbone.vit import get_vit_lr_decay_rate
from ape.modeling.backbone.vit_eva02 import SimpleFeaturePyramid, ViT
from ape.modeling.text import EVA01CLIP

from .....detrex.detectron2.configs.common.data.constants import constants
from ...common.data.coco_refcoco_instance_lsj1024_wtags import dataloader
from .ape_deta_r50_12ep import model

model.model_vision.pixel_mean = constants.imagenet_rgb256_mean
model.model_vision.pixel_std = constants.imagenet_rgb256_std
model.model_vision.input_format = "RGB"

model.model_vision.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        window_size=16,
        mlp_ratio=4 * 2 / 3,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=list(range(0, 5))
        + list(range(6, 11))
        + list(range(12, 17))
        + list(range(18, 23)),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        use_act_checkpoint=True,
        xattn=True,
        swiglu=True,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.model_vision.neck = None

model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

optimizer = get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.8, num_layers=24)
    if "backbone.net" in module_name
    else 1
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.params.weight_decay_norm = None

optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

train = get_config("common/train.py").train
train.max_iter = 10000
train.eval_period = 2000
train.log_period = 50

train.checkpointer.period = 2000
train.checkpointer.max_to_keep = 2

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"

train.init_checkpoint = (
    "models/Yuxin-CV/EVA-02/eva02/pt/eva02_L_pt_in21k_p14to16.pt?matching_heuristics=True"
)

train.amp.enabled = True
train.ddp.fp16_compression = True

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [75000, 90000]
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 8
dataloader.train.total_batch_size_list = [8, 8]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

model.model_vision.dataset_prompts = ["name", "expression"]
model.model_vision.dataset_names = ["coco_2017", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.output_dir = train.output_dir

model.model_language = L(EVA01CLIP)(
    clip_model="EVA_CLIP_g_14_X", cache_dir="models/BAAI/EVA/eva_clip_psz14.pt"
)
model.model_vision.embed_dim_language = 1024

