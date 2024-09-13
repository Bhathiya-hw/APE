from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.config import get_config
from ape.modeling.backbone.vit import get_vit_lr_decay_rate
from ape.modeling.backbone.vit_eva02 import SimpleFeaturePyramid, ViT

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.solver import WarmupParamScheduler
from detrex.modeling.neck import ChannelMapper
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ape.data.detection_utils import get_fed_loss_cls_weights
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
)

from ape.modeling.text import EVA02CLIP

from .....detrex.detectron2.configs.common.data.constants import constants
from ...common.data.coco_refcoco_instance_lsj1024_wtags import dataloader
from .ape_deta_r50_12ep import model

from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_24ep import (
    model,
    optimizer,
    train,
)
from ...common.backbone.vitt_eva02 import backbone

model.model_vision.pixel_mean = constants.imagenet_rgb256_mean
model.model_vision.pixel_std = constants.imagenet_rgb256_std
model.model_vision.input_format = "RGB"

# model.model_vision.backbone = L(SimpleFeaturePyramid)(
#     net=L(ViT)(  # Single-scale ViT backbone
#         img_size=1024,
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         drop_path_rate=0.4,
#         window_size=16,
#         mlp_ratio=4 * 2 / 3,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         window_block_indexes=list(range(0, 5))
#         + list(range(6, 11))
#         + list(range(12, 17))
#         + list(range(18, 23)),
#         residual_block_indexes=[],
#         use_rel_pos=True,
#         out_feature="last_feat",
#         use_act_checkpoint=True,
#         xattn=True,
#         swiglu=True,
#     ),
#     in_feature="${.net.out_feature}",
#     out_channels=256,
#     scale_factors=(4.0, 2.0, 1.0, 0.5),
#     top_block=L(LastLevelMaxPool)(),
#     norm="LN",
#     square_pad=1024,
# )

# model.model_vision.neck = None

# model.model_vision.mask_in_features = ["p2"]
# model.model_vision.input_shapes = {
#     "p2": ShapeSpec(channels=256),
#     "p3": ShapeSpec(channels=256),
#     "p4": ShapeSpec(channels=256),
#     "p5": ShapeSpec(channels=256),
#     "p6": ShapeSpec(channels=256),
# }

model.model_vision.backbone = backbone
model.model_vision.embed_dim_language = 1024

model.model_vision.neck = L(ChannelMapper)(
    input_shapes={
        "p2": ShapeSpec(channels=256),
        "p3": ShapeSpec(channels=256),
        "p4": ShapeSpec(channels=256),
        "p5": ShapeSpec(channels=256),
        "p6": ShapeSpec(channels=256),
    },
    in_features=["p2", "p3", "p4", "p5", "p6"],
    out_channels=256,
    num_outs=5,
    kernel_size=1,
    norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
)

model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

model.model_vision.transformer.encoder.num_layers = 6
model.model_vision.transformer.decoder.num_layers = 6
model.model_vision.transformer.encoder.embed_dim = 256
model.model_vision.transformer.decoder.embed_dim = 256
model.model_vision.embed_dim = 256
model.model_vision.backbone.out_channels = 256

model.model_vision.update(
    _target_=DeformableDETRSegmVL,
)
model.model_vision.transformer.update(
    _target_=DeformableDetrTransformerVL,
)
model.model_vision.transformer.encoder.update(
    _target_=DeformableDetrTransformerEncoderVL,
)
model.model_vision.transformer.decoder.update(
    _target_=DeformableDetrTransformerDecoderVL,
)

model.model_vision.transformer.encoder.vl_layer = L(VisionLanguageFusion)(
    v_dim="${....embed_dim}",
    l_dim="${....embed_dim_language}",
    embed_dim=2048,
    num_heads=8,
    dropout=0.1,
    drop_path=0.0,
    init_values=1.0 / 6,
    stable_softmax_2d=True,
    clamp_min_for_underflow=True,
    clamp_max_for_overflow=True,
    use_checkpoint=True,
)
model.model_vision.transformer.encoder.use_act_checkpoint = True

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True
model.model_vision.name_prompt_fusion_type = "zero"

model.model_vision.num_classes = 1256
model.model_vision.select_box_nums_for_evaluation = 300

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
train.max_iter = 1000
train.eval_period = 2000
train.log_period = 20

train.checkpointer.period = 500
train.checkpointer.max_to_keep = 2

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"

# train.init_checkpoint = (
#     "models/Yuxin-CV/EVA-02/eva02/pt/eva02_L_pt_in21k_p14to16.pt?matching_heuristics=True"
# )

train.init_checkpoint= ("/mnt/storage/checkpoints/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth")
train.amp.enabled = True
train.ddp.fp16_compression = True

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [750, 900]
lr_multiplier.warmup_length = 100 / train.max_iter

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 4
dataloader.train.total_batch_size_list = [4, 4]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

model.model_vision.dataset_prompts = ["expression"]
model.model_vision.dataset_names = ["refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.output_dir = train.output_dir

# model.model_language = L(EVA01CLIP)(
#     clip_model="EVA_CLIP_g_14_X", cache_dir="models/BAAI/EVA/eva_clip_psz14.pt"
# )

model.model_language = L(EVA02CLIP)(
    clip_model="EVA02-CLIP-bigE-14-plus",
    cache_dir="models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt",
    dtype="float16",
)

model.model_vision.embed_dim_language = 1024