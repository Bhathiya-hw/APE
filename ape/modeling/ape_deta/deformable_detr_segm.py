import copy
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from ape.modeling.text import utils as text_utils
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import Conv2d, ShapeSpec, get_norm, move_device_like
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms

from .deformable_detr import DeformableDETR
from .fast_rcnn import fast_rcnn_inference
from .segmentation import MaskHeadSmallConv, MHAttentionMap


class DeformableDETRSegm(DeformableDETR):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        instance_on: bool = True,
        semantic_on: bool = False,
        panoptic_on: bool = False,
        freeze_detr=False,
        input_shapes=[],
        mask_in_features=[],
        mask_encode_level=0,
        stuff_dataset_learn_thing: bool = True,
        stuff_prob_thing: float = -1.0,
        name_prompt_fusion_type: str = "none",
        name_prompt_fusion_text: bool = None,
        test_mask_on: bool = True,
        semantic_post_nms: bool = True,
        panoptic_post_nms: bool = True,
        aux_mask: bool = False,
        panoptic_configs: dict = {
            "prob": 0.1,
            "pano_temp": 0.06,
            "transform_eval": True,
            "object_mask_threshold": 0.01,
            "overlap_threshold": 0.4,
        },
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.instance_on = instance_on
        self.semantic_on = semantic_on
        self.panoptic_on = panoptic_on

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        self.input_shapes = input_shapes
        self.mask_in_features = mask_in_features
        self.mask_encode_level = mask_encode_level

        hidden_dim = self.transformer.embed_dim
        norm = "GN"
        use_bias = False

        assert len(self.mask_in_features) == 1
        in_channels = [self.input_shapes[feat_name].channels for feat_name in self.mask_in_features]
        in_channel = in_channels[0]

        self.lateral_conv = Conv2d(
            in_channel,
            hidden_dim,
            kernel_size=1,
            stride=1,
            bias=use_bias,
            padding=0,
            norm=get_norm(norm, hidden_dim),
        )
        self.output_conv = Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            bias=use_bias,
            padding=1,
            norm=get_norm(norm, hidden_dim),
            activation=F.relu,
        )
        self.mask_conv = Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, bias=use_bias, padding=0
        )

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.aux_mask = aux_mask
        if self.aux_mask:
            self.mask_embed = nn.ModuleList(
                [copy.deepcopy(self.mask_embed) for i in range(len(self.class_embed) - 1)]
            )

        weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)
        weight_init.c2_xavier_fill(self.mask_conv)

        self.stuff_dataset_learn_thing = stuff_dataset_learn_thing
        self.stuff_prob_thing = stuff_prob_thing
        self.test_mask_on = test_mask_on
        self.semantic_post_nms = semantic_post_nms
        self.panoptic_post_nms = panoptic_post_nms
        self.panoptic_configs = panoptic_configs

        self.name_prompt_fusion_type = name_prompt_fusion_type
        self.name_prompt_fusion_text = name_prompt_fusion_text
        if name_prompt_fusion_type == "learnable":
            self.name_prompt_fusion_feature = nn.Parameter(
                torch.Tensor(1, 1, self.embed_dim_language)
            )
            nn.init.normal_(self.name_prompt_fusion_feature)
        elif name_prompt_fusion_type == "zero":
            self.name_prompt_fusion_feature = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim_language), requires_grad=False
            )
        else:
            self.name_prompt_fusion_feature = None

    def forward(self, batched_inputs, do_postprocess=True):
        if self.training:
            if "dataset_id" in batched_inputs[0]:
                dataset_ids = [x["dataset_id"] for x in batched_inputs]
                assert len(set(dataset_ids)) == 1, dataset_ids
                dataset_id = dataset_ids[0]
            else:
                dataset_id = 0
        else:
            dataset_id = self.eval_dataset_id

        if dataset_id >= 0:
            prompt = self.dataset_prompts[dataset_id]
        elif "prompt" in batched_inputs[0]:
            prompt = batched_inputs[0]["prompt"]
        else:
            prompt = "name"

        if prompt == "expression":
            for x in batched_inputs:
                if isinstance(x["expressions"], List):
                    pass
                else:
                    x["expressions"] = [x["expressions"]]
                if 'expression_tags' in x:
                    if isinstance(x['expression_tags'][0], List):
                        pass
                    else:
                        x['expression_tags'] = [x['expression_tags']]
                assert all([len(xx) > 0 for xx in x["expressions"]])
                assert all([isinstance(xx, str) for xx in x["expressions"]])
                self.test_topk_per_image = 1
        else:
            self.test_topk_per_image = self.select_box_nums_for_evaluation
        if self.select_box_nums_for_evaluation_list is not None:
            self.test_topk_per_image = self.select_box_nums_for_evaluation_list[dataset_id]

        if self.training and prompt == "phrase":
            gt_num = torch.tensor([len(input["instances"]) for input in batched_inputs]).to(
                self.device
            )
            gt_classes = torch.arange(gt_num.sum()).to(self.device)
            gt_cumsum = torch.cumsum(gt_num, dim=0).to(self.device)
            for i, input in enumerate(batched_inputs):
                if i == 0:
                    input["instances"].gt_classes = gt_classes[: gt_cumsum[i]]
                else:
                    input["instances"].gt_classes = gt_classes[gt_cumsum[i - 1] : gt_cumsum[i]]
        if self.training and prompt == "expression":
            gt_num = torch.tensor([len(input["instances"]) for input in batched_inputs]).to(
                self.device
            )
            gt_classes = torch.arange(gt_num.sum()).to(self.device)
            gt_cumsum = torch.cumsum(gt_num, dim=0).to(self.device)
            for i, input in enumerate(batched_inputs):
                if i == 0:
                    input["instances"].gt_classes = gt_classes[: gt_cumsum[i]]
                else:
                    input["instances"].gt_classes = gt_classes[gt_cumsum[i - 1] : gt_cumsum[i]]

                if not self.expression_cumulative_gt_class:
                    input["instances"].gt_classes *= 0

        if prompt == "text":
            texts = [x["text_prompt"] for x in batched_inputs]
            text_promp_text_list = [x.strip() for x in ",".join(texts).split(",")]
            text_promp_text_list = [x for x in text_promp_text_list if len(x) > 0]

            if any([True if x.count(" ") >= 1 else False for x in text_promp_text_list]):
                prompt = "phrase"
            else:
                prompt = "name"
        else:
            text_promp_text_list = None

        text_tags_list = None
        if prompt == "name":
            if text_promp_text_list:
                text_list = text_promp_text_list
                cache = False
            elif dataset_id >= 0:
                text_list = get_text_list(
                    self.metadata_list[dataset_id], self.dataset_entities[dataset_id]
                )
                cache = True
            else:
                text_list = []
                for metadata, dataset_entity in zip(self.metadata_list, self.dataset_entities):
                    text_list += get_text_list(metadata, dataset_entity)
                text_list = text_list[:1203+365+601]
                text_list = text_list[:1203]
                cache = True

                # from detectron2.data.catalog import MetadataCatalog
                # metadata = MetadataCatalog.get("coco_2017_train_panoptic_separated")
                # text_list = get_text_list(metadata, "thing+stuff")

            outputs_l = self.model_language.forward_text(text_list, cache=cache)
            if "last_hidden_state_eot" in outputs_l:
                features_l = outputs_l["last_hidden_state_eot"]
            else:
                features_l = text_utils.reduce_language_feature(
                    outputs_l["last_hidden_state"],
                    outputs_l["attention_mask"],
                    reduce_type=self.text_feature_reduce_type,
                )
            attention_mask_l = None

            if (
                dataset_id >= 0
                and self.dataset_entities[dataset_id] == "stuff"
                and self.metadata_list[dataset_id].get("stuff_classes")[0] == "things"
                and not self.stuff_dataset_learn_thing
            ):
                features_l[0, :] *= 0
                if self.training:
                    for i, input in enumerate(batched_inputs):
                        input["instances"] = input["instances"][input["instances"].gt_classes > 0]

            if self.text_feature_batch_repeat or True:
                features_l = features_l.unsqueeze(0).repeat(len(batched_inputs), 1, 1)
            else:
                features_l = features_l.unsqueeze(1)

        elif prompt == "phrase" or prompt == "expression":
            if text_promp_text_list:
                text_list = text_promp_text_list
            elif prompt == "phrase":
                text_list = [phrase for x in batched_inputs for phrase in x["instances"].phrases]
            elif prompt == "expression":
                text_list = [xx for x in batched_inputs for xx in x["expressions"]]
                if 'expression_tags' in batched_inputs:
                    text_tags_list = ["|".join(xx) for x in batched_inputs for xx in x['expression_tags']] 


            outputs_l = self.model_language.forward_text(text_list)
            outputs_tl = None
            if 'expression_tags' in batched_inputs:
                outputs_tl = self.model_language.forward_text(text_tags_list)
            if self.text_feature_reduce_before_fusion:
                if "last_hidden_state_eot" in outputs_l:
                    features_l = outputs_l["last_hidden_state_eot"]
                else:
                    features_l = text_utils.reduce_language_feature(
                        outputs_l["last_hidden_state"],
                        outputs_l["attention_mask"],
                        reduce_type=self.text_feature_reduce_type,
                    )
                if outputs_tl:
                    if "last_hidden_state_eot" in outputs_tl:
                        features_tl = outputs_tl["last_hidden_state_eot"]
                    else:
                        features_tl = text_utils.reduce_language_feature(
                            outputs_tl["last_hidden_state"],
                            outputs_tl["attention_mask"],
                            reduce_type=self.text_feature_reduce_type,
                        )                
                    features_l = torch.sum([features_l, features_tl], dim=0)
                attention_mask_l = None
                if (
                    self.text_feature_bank
                    and not self.text_feature_bank_reset
                    and dataset_id >= 0
                    and dataset_id < len(self.metadata_list)
                ):
                    features_l = torch.cat(
                        [features_l, self.features_phrase_bank[dataset_id]], dim=0
                    )
                    features_l = features_l[
                        : max(len(text_list), self.criterion[dataset_id].num_classes)
                    ]
                    self.features_phrase_bank[
                        dataset_id, : self.criterion[dataset_id].num_classes
                    ] = features_l[: self.criterion[dataset_id].num_classes]
                elif self.text_feature_bank and self.text_feature_bank_reset:
                    features_l = torch.cat(
                        [features_l, self.features_phrase_bank[dataset_id] * 0], dim=0
                    )
                    features_l = features_l[
                        : max(len(text_list), self.criterion[dataset_id].num_classes)
                    ]

                if self.text_feature_bank and self.text_feature_bank_random_size:
                    features_l = features_l[
                        : random.randint(len(text_list), len(features_l))
                    ]

                if self.text_feature_batch_repeat:
                    features_l = features_l.unsqueeze(0).repeat(len(batched_inputs), 1, 1)
                else:
                    features_l = features_l.unsqueeze(1)
            else:
                features_l = outputs_l["last_hidden_state"]
                attention_mask_l = outputs_l["attention_mask"]

        start_time = time.perf_counter()
        images = self.preprocess_image(batched_inputs)

        batch_size, _, H, W = images.tensor.shape
        img_masks = images.tensor.new_ones(batch_size, H, W)
        for image_id, image_size in enumerate(images.image_sizes):
            img_masks[image_id, : image_size[0], : image_size[1]] = 0
        self.preprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        features = self.backbone(images.tensor)  # output feature dict
        self.backbone_time = time.perf_counter() - start_time

        if self.neck is not None:
            multi_level_feats = self.neck({f: features[f] for f in self.neck.in_features})
        else:
            multi_level_feats = [feat for feat_name, feat in features.items()]
        multi_level_masks = []
        multi_level_position_embeddings = []
        spatial_shapes = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1]).to(images.tensor.dtype)
            )

            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        start_time = time.perf_counter()
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            anchors,
            memory,
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds
        )
        self.transformer_time = time.perf_counter() - start_time

        mask_features = self.maskdino_mask_features(memory, features, multi_level_masks)

        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if prompt == "name":
                outputs_class = self.class_embed[lvl](inter_states[lvl], features_l)
            elif prompt == "phrase" or prompt == "expression":
                outputs_class = self.class_embed[lvl](inter_states[lvl], features_l)
            else:
                outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            if self.aux_mask:
                mask_embeds = self.mask_embed[lvl](inter_states[lvl])
            else:
                mask_embeds = self.mask_embed(inter_states[lvl])
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
            outputs_masks.append(outputs_mask)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks
        outputs_mask[-1] += 0.0 * sum(outputs_mask)

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": outputs_mask[-1],
            "init_reference": init_reference,
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_mask,
            )

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "anchors": anchors,
                "spatial_shapes": spatial_shapes,
                "image_tensor_size": images.tensor.size()[2:],
            }

        if (
            self.vis_period > 0
            and self.training
            and get_event_storage().iter % self.vis_period == self.vis_period - 1
        ):
            self.visualize_training(batched_inputs, output, images, dataset_id)
            self.visualize_training_enc_output(batched_inputs, output, images, dataset_id)
            self.visualize_training_enc_output_nonms(batched_inputs, output, images, dataset_id)
            self.visualize_training_init_reference(batched_inputs, output, images, dataset_id)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)

            if (
                self.vis_period > 0
                and self.training
                and get_event_storage().iter % self.vis_period == self.vis_period - 1
            ):
                enc_outputs = output["enc_outputs"]
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
                if self.criterion[dataset_id].matcher_stage1 is not None:
                    tmp1 = self.criterion[dataset_id].matcher_stage1.positive_fraction
                    tmp2 = self.criterion[dataset_id].matcher_stage1.batch_size_per_image
                    self.criterion[dataset_id].matcher_stage1.positive_fraction = 1.0
                    self.criterion[dataset_id].matcher_stage1.batch_size_per_image = 5120000
                    indices, ious = self.criterion[dataset_id].matcher_stage1(
                        enc_outputs, bin_targets, return_cost_matrix=True
                    )
                    self.criterion[dataset_id].matcher_stage1.positive_fraction = tmp1
                    self.criterion[dataset_id].matcher_stage1.batch_size_per_image = tmp2

                    self.visualize_training_enc_output_pos(
                        batched_inputs, output, images, dataset_id, indices
                    )

                if self.criterion[dataset_id].matcher_stage2 is not None:
                    indices = self.criterion[dataset_id].matcher_stage2(output, targets)

                    self.visualize_training_init_reference_pos(
                        batched_inputs, output, images, dataset_id, indices
                    )

            loss_dict = self.criterion[dataset_id](output, targets)

            weight_dict = self.criterion[dataset_id].weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:

            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"]

            start_time = time.perf_counter()

            iter_func = retry_if_cuda_oom(F.interpolate)
            mask_pred = iter_func(
                mask_pred, size=images.tensor.size()[2:], mode="bilinear", align_corners=False
            )

            merged_results = [{} for _ in range(box_cls.size(0))]
            if self.instance_on and not (
                self.eval_dataset_entity and "thing" not in self.eval_dataset_entity
            ):
                if dataset_id >= 0 and dataset_id < len(self.metadata_list):
                    if is_thing_stuff_overlap(self.metadata_list[dataset_id]):
                        thing_id = self.metadata_list[
                            dataset_id
                        ].thing_dataset_id_to_contiguous_id.values()
                        thing_id = torch.Tensor(list(thing_id)).to(torch.long).to(self.device)

                        detector_box_cls = torch.zeros_like(box_cls)
                        detector_box_cls += float("-inf")
                        detector_box_cls[..., thing_id] = box_cls[..., thing_id]
                    else:
                        num_thing_classes = len(self.metadata_list[dataset_id].thing_classes)
                        detector_box_cls = box_cls[..., :num_thing_classes]
                else:
                    detector_box_cls = box_cls

                use_sigmoid = True
                detector_results, filter_inds = self.inference(
                    detector_box_cls, box_pred, images.image_sizes, use_sigmoid=use_sigmoid
                )

                if self.test_mask_on:
                    detector_mask_preds = [
                        x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                    ]

                    for result, box_mask in zip(detector_results, detector_mask_preds):
                        box_mask = box_mask.sigmoid() > 0.5
                        box_mask = BitMasks(box_mask).crop_and_resize(
                            result.pred_boxes.tensor.to(box_mask.device), 128
                        )
                        result.pred_masks = (
                            box_mask.to(result.pred_boxes.tensor.device)
                            .unsqueeze(1)
                            .to(dtype=torch.float32)
                        )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    detector_results = DeformableDETRSegm._postprocess_instance(
                        detector_results, batched_inputs, images.image_sizes
                    )
                    for merged_result, detector_result in zip(merged_results, detector_results):
                        merged_result.update(detector_result)

            else:
                detector_results = None

            if self.semantic_on and not (
                self.eval_dataset_entity and "stuff" not in self.eval_dataset_entity
            ):

                semantic_mask_pred = mask_pred.clone()
                semantic_box_cls = get_stuff_score(
                    box_cls, self.metadata_list[dataset_id], self.dataset_entities[dataset_id]
                )

                if self.semantic_post_nms:
                    _, filter_inds = self.inference(semantic_box_cls, box_pred, images.image_sizes)
                    semantic_box_cls = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_box_cls, filter_inds)],
                        dim=0,
                    )
                    semantic_mask_pred = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_mask_pred, filter_inds)],
                        dim=0,
                    )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    semantic_results = DeformableDETRSegm._postprocess_semantic(
                        semantic_box_cls, semantic_mask_pred, batched_inputs, images
                    )
                    if (
                        dataset_id >= 0
                        and self.dataset_entities[dataset_id] == "stuff"
                        and self.metadata_list[dataset_id].get("stuff_classes")[0] == "things"
                        and self.stuff_prob_thing > 0
                    ):
                        for semantic_result in semantic_results:
                            semantic_result["sem_seg"][0, ...] = math.log(
                                self.stuff_prob_thing / (1 - self.stuff_prob_thing)
                            )
                    for merged_result, semantic_result in zip(merged_results, semantic_results):
                        merged_result.update(semantic_result)

            else:
                semantic_results = None

            if self.panoptic_on and not (
                self.eval_dataset_entity and "thing+stuff" not in self.eval_dataset_entity
            ):
                assert dataset_id >= 0 and dataset_id < len(self.metadata_list)
                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    if True:
                        if self.panoptic_post_nms:
                            _, filter_inds = self.inference(box_cls, box_pred, images.image_sizes)
                            panoptic_mask_pred = [
                                x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                            ]
                            panoptic_box_cls = [
                                x[filter_ind] for x, filter_ind in zip(box_cls, filter_inds)
                            ]

                        panoptic_results = DeformableDETRSegm._postprocess_panoptic(
                            panoptic_box_cls,
                            panoptic_mask_pred,
                            batched_inputs,
                            images,
                            self.metadata_list[dataset_id],
                            self.panoptic_configs,
                        )
                    else:
                        panoptic_results = []
                        self.combine_overlap_thresh = 0.5
                        self.combine_stuff_area_thresh = 4096
                        self.combine_instances_score_thresh = 0.5
                        for detector_result, semantic_result in zip(
                            detector_results, semantic_results
                        ):
                            detector_r = detector_result["instances"]
                            sem_seg_r = semantic_result["sem_seg"]
                            panoptic_r = combine_semantic_and_instance_outputs(
                                detector_r,
                                sem_seg_r.argmax(dim=0),
                                self.combine_overlap_thresh,
                                self.combine_stuff_area_thresh,
                                self.combine_instances_score_thresh,
                            )
                            panoptic_results.append({"panoptic_seg": panoptic_r})
                    for merged_result, panoptic_result in zip(merged_results, panoptic_results):
                        merged_result.update(panoptic_result)

            else:
                panoptic_results = None

            self.postprocess_time = time.perf_counter() - start_time

            if do_postprocess:
                return merged_results

            return detector_results, semantic_results, panoptic_results

    def maskdino_mask_features(self, encode_feats, multi_level_feats, multi_level_masks):
        start_idx = sum(
            [mask.shape[1] * mask.shape[2] for mask in multi_level_masks[: self.mask_encode_level]]
        )
        end_idx = sum(
            [
                mask.shape[1] * mask.shape[2]
                for mask in multi_level_masks[: self.mask_encode_level + 1]
            ]
        )
        b, h, w = multi_level_masks[self.mask_encode_level].size()

        encode_feats = encode_feats[:, start_idx:end_idx, :]
        encode_feats = encode_feats.permute(0, 2, 1).reshape(b, -1, h, w)

        x = [multi_level_feats[f] for f in self.mask_in_features]
        x = x[0]
        x = self.lateral_conv(x)
        x = x + F.interpolate(encode_feats, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.output_conv(x)
        mask_features = self.mask_conv(x)

        return mask_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes, use_sigmoid=True):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """

        if True:

            if use_sigmoid:
                scores = torch.cat(
                    (
                        box_cls.sigmoid(),
                        torch.zeros((box_cls.size(0), box_cls.size(1), 1), device=self.device),
                    ),
                    dim=2,
                )
            else:
                scores = torch.cat(
                    (
                        box_cls,
                        torch.zeros((box_cls.size(0), box_cls.size(1), 1), device=self.device),
                    ),
                    dim=2,
                )

            boxes = box_cxcywh_to_xyxy(box_pred)

            img_h = torch.tensor([image_size[0] for image_size in image_sizes], device=self.device)
            img_w = torch.tensor([image_size[1] for image_size in image_sizes], device=self.device)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            boxes = boxes.unbind(0)
            scores = scores.unbind(0)
            image_shapes = image_sizes

            results, filter_inds = fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
            )

            return results, filter_inds

        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results, topk_indexes

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            if not targets_per_image.has("gt_masks"):
                gt_masks = torch.zeros((0, h, w), dtype=torch.bool)
            else:
                gt_masks = targets_per_image.gt_masks

            if not isinstance(gt_masks, torch.Tensor):
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    gt_masks = BitMasks.from_polygon_masks(gt_masks, h, w).tensor

            gt_masks = self._move_to_current_device(gt_masks)
            gt_masks = ImageList.from_tensors(
                [gt_masks],
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,
            ).tensor.squeeze(0)

            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks": gt_masks})

            if targets_per_image.has("is_thing"):
                new_targets[-1]["is_thing"] = targets_per_image.is_thing

        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [x.to(self.pixel_mean.dtype) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess_instance(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r.to("cpu")})
        return processed_results

    @staticmethod
    def _postprocess_semantic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        pano_temp=0.06,
        transform_eval=True,
    ):
        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            T = pano_temp
            mask_cls = mask_cls.sigmoid()

            if transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            if mask_cls.size(1) > 1000:
                mask_cls = mask_cls.cpu()
                mask_pred = mask_pred.cpu()
            result = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results

    @staticmethod
    def _postprocess_panoptic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        metadata,
        panoptic_configs,
    ):
        prob = panoptic_configs["prob"]
        pano_temp = panoptic_configs["pano_temp"]
        transform_eval = panoptic_configs["transform_eval"]
        object_mask_threshold = panoptic_configs["object_mask_threshold"]
        overlap_threshold = panoptic_configs["overlap_threshold"]

        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            mask_pred = sem_seg_postprocess(mask_pred, image_size, height, width)

            T = pano_temp
            scores, labels = mask_cls.sigmoid().max(-1)
            mask_pred = mask_pred.sigmoid()
            keep = scores > object_mask_threshold
            if transform_eval:
                scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            panoptic_seg = torch.zeros((height, width), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.size(0) > 0:

                cur_mask_ids = cur_prob_masks.argmax(0)

            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    if not isthing and metadata.get("stuff_classes")[0] == "things":
                        pred_class = int(pred_class) - len(metadata.thing_classes) + 1

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            processed_results.append({"panoptic_seg": (panoptic_seg, segments_info)})
        return processed_results

    @torch.no_grad()
    def visualize_training(
        self, batched_inputs, output, images, dataset_id, suffix="", do_nms=True
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        pred_masks = output["pred_masks"]

        thing_classes = self.metadata_list[dataset_id].get("thing_classes", [])
        stuff_classes = self.metadata_list[dataset_id].get("stuff_classes", [])
        if len(thing_classes) > 0 and len(stuff_classes) > 0 and stuff_classes[0] == "things":
            stuff_classes = stuff_classes[1:]
        if is_thing_stuff_overlap(self.metadata_list[dataset_id]):
            class_names = (
                thing_classes if len(thing_classes) > len(stuff_classes) else stuff_classes
            )
        else:
            class_names = thing_classes + stuff_classes

        if "instances" in batched_inputs[0] and batched_inputs[0]["instances"].has("phrases"):
            class_names = [phrase for x in batched_inputs for phrase in x["instances"].phrases] + [
                "unknown"
            ] * 1000
        if "expressions" in batched_inputs[0] and self.expression_cumulative_gt_class:
            class_names = [x["expressions"] for x in batched_inputs] + ["unknown"] * 1000

        num_thing_classes = len(class_names)
        pred_logits = pred_logits[..., :num_thing_classes]

        if pred_masks is not None:
            pred_masks = [
                F.interpolate(
                    pred_mask.float().cpu().unsqueeze(0),
                    size=images.tensor.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if pred_mask.size(0) > 0
                else pred_mask
                for pred_mask in pred_masks
            ]
        else:
            pred_masks = [
                torch.zeros(pred_box.size(0), image_size[0], image_size[1])
                for pred_box, image_size in zip(pred_boxes, images.image_sizes)
            ]

        if do_nms:
            results, filter_inds = self.inference(pred_logits, pred_boxes, images.image_sizes)
            pred_masks = [
                pred_mask[filter_ind.cpu()]
                for pred_mask, filter_ind in zip(pred_masks, filter_inds)
            ]
            for result, pred_mask in zip(results, pred_masks):
                result.pred_masks = pred_mask.sigmoid() > 0.5
        else:
            results = []
            for pred_logit, pred_box, pred_mask, image_size in zip(
                pred_logits, pred_boxes, pred_masks, images.image_sizes
            ):
                result = Instances(image_size)
                result.pred_boxes = Boxes(pred_box)
                result.scores = pred_logit[:, 0]
                result.pred_classes = torch.zeros(
                    len(pred_box), dtype=torch.int64, device=pred_logit.device
                )
                result.pred_masks = pred_mask.sigmoid() > 0.5

                results.append(result)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):

            if "expressions" in batched_inputs[0] and not self.expression_cumulative_gt_class:
                class_names = [input["expressions"]] + ["unknown"] * 1000

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)

            if "instances" in input:
                labels = [
                    "{}".format(class_names[gt_class]) for gt_class in input["instances"].gt_classes
                ]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            labels = [
                "{}_{:.0f}%".format(class_names[pred_class], score * 100)
                for pred_class, score in zip(result.pred_classes.cpu(), result.scores.cpu())
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                labels=labels,
                masks=result.pred_masks[:, : img.shape[0], : img.shape[1]]
                .clone()
                .detach()
                .cpu()
                .numpy()
                if result.has("pred_masks")
                else None,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            if result.has("pred_texts"):
                labels = [
                    "{}".format(text) for text, score in zip(result.pred_texts, result.scores.cpu())
                ]
                v_pred = Visualizer(img, None)
                v_pred = v_pred.overlay_instances(
                    boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                    labels=labels,
                    masks=result.pred_masks.clone().detach().cpu().numpy(),
                )
                pred_img = v_pred.get_image()
                vis_img = np.concatenate((vis_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "training", str(storage.iter) + suffix + "_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", suffix + basename),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_inference_panoptic(self, batched_inputs, results, dataset_id):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            height = input["height"]
            width = input["width"]
            img = cv2.resize(img, (width, height))

            v_gt = Visualizer(img, self.metadata_list[dataset_id])

            if "instances" in input:
                labels = [
                    "{}".format(class_names[gt_class]) for gt_class in input["instances"].gt_classes
                ]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            v_pred = Visualizer(img, self.metadata_list[dataset_id])

            panoptic_seg, segments_info = result["panoptic_seg"]
            v_pred = v_pred.draw_panoptic_seg_predictions(panoptic_seg.cpu(), segments_info)
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "training", str(storage.iter) + "_pan_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", "pan_" + basename),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_training_enc_output(self, batched_inputs, output, images, dataset_id, suffix=""):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["enc_outputs"]["pred_logits"]
        pred_boxes = output["enc_outputs"]["pred_boxes"]

        results, filter_inds = self.inference(pred_logits, pred_boxes, images.image_sizes)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            labels = [
                "{}_{:.0f}%".format(pred_class, score * 100)
                for pred_class, score in zip(result.pred_classes.cpu(), result.scores.cpu())
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                labels=labels,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_enc_output_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", suffix + "enc_output_" + basename),
                    vis_img[:, :, ::-1],
                )

    def visualize_training_enc_output_nonms(
        self, batched_inputs, output, images, dataset_id, suffix=""
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["enc_outputs"]["pred_logits"]
        pred_boxes = output["enc_outputs"]["pred_boxes"]

        image_sizes = images.image_sizes
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

        img_h = torch.tensor([image_size[0] for image_size in image_sizes], device=self.device)
        img_w = torch.tensor([image_size[1] for image_size in image_sizes], device=self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = pred_boxes * scale_fct[:, None, :]

        pred_boxes = pred_boxes.unbind(0)
        pred_logits = pred_logits.unbind(0)

        from detectron2.utils.visualizer import Visualizer

        for input, pred_box, pred_logit in zip(batched_inputs, pred_boxes, pred_logits):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            keep = pred_logit.sigmoid() > 0.1
            if keep.sum() == 0:
                continue
            pred_box = pred_box[keep.squeeze()]
            pred_logit = pred_logit[keep.squeeze()]

            labels = [
                "{:.0f}%".format(score * 100) for score in pred_logit.squeeze().cpu().tolist()
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=pred_box.clone().detach().cpu().numpy(),
                labels=labels,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_enc_output_nonms_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "inference", suffix + "enc_output_nonms_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_training_init_reference(
        self, batched_inputs, output, images, dataset_id, suffix=""
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_boxes = output["init_reference"]

        image_sizes = images.image_sizes
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

        img_h = torch.tensor([image_size[0] for image_size in image_sizes], device=self.device)
        img_w = torch.tensor([image_size[1] for image_size in image_sizes], device=self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = pred_boxes * scale_fct[:, None, :]

        pred_boxes = pred_boxes.unbind(0)

        from detectron2.utils.visualizer import Visualizer

        for input, pred_box in zip(batched_inputs, pred_boxes):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=pred_box.clone().detach().cpu().numpy(),
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_init_reference_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "inference", suffix + "init_reference_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_training_enc_output_pos(
        self, batched_inputs, output, images, dataset_id, indices, suffix=""
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        anchors = output["enc_outputs"]["anchors"]

        image_sizes = images.image_sizes
        anchors = box_cxcywh_to_xyxy(anchors)

        img_h, img_w = torch.tensor(image_sizes, device=self.device).unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        anchors = anchors * scale_fct[:, None, :]

        anchors = anchors.unbind(0)

        from detectron2.utils.visualizer import Visualizer

        for input, anchor, indice in zip(batched_inputs, anchors, indices):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                labels = ["{}".format(idx) for idx in range(len(input["instances"]))]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=anchor.clone().detach().cpu().numpy(),
            )
            pred_img = v_pred.get_image()

            anchor = anchor[indice[0], :]
            labels = ["{}".format(idx) for idx in indice[1]]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=anchor.clone().detach().cpu().numpy(),
                labels=labels,
            )
            pred_img2 = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img, pred_img2), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_enc_output_pos_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "inference", suffix + "enc_output_pos_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_training_init_reference_pos(
        self, batched_inputs, output, images, dataset_id, indices, suffix=""
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_boxes = output["init_reference"]

        image_sizes = images.image_sizes
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

        img_h = torch.tensor([image_size[0] for image_size in image_sizes], device=self.device)
        img_w = torch.tensor([image_size[1] for image_size in image_sizes], device=self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = pred_boxes * scale_fct[:, None, :]

        pred_boxes = pred_boxes.unbind(0)

        from detectron2.utils.visualizer import Visualizer

        for input, pred_box, indice in zip(batched_inputs, pred_boxes, indices):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                labels = ["{}".format(idx) for idx in range(len(input["instances"]))]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            pred_box = pred_box[indice[0]]
            labels = ["{}".format(idx) for idx in indice[1]]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=pred_box.clone().detach().cpu().numpy(),
                labels=labels,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_init_reference_pos_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "inference", suffix + "init_reference_pos_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )

    def set_model_language(self, model_language):
        self.model_language = model_language


class NMSPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, select_box_nums_for_evaluation):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_mask = outputs["pred_masks"]
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = torch.div(all_indexes, out_logits.shape[2], rounding_mode="trunc")
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        keep_inds_all = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]
            mask = out_mask[b]

            pre_topk = score.topk(10000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.7)[:select_box_nums_for_evaluation]

            result = Instances(target_sizes[b])
            result.pred_boxes = Boxes(box[keep_inds])
            result.scores = score[keep_inds]
            result.pred_classes = lbls[keep_inds]
            results.append(result)

            keep_inds_all.append(keep_inds)

        return results, keep_inds_all


def is_thing_stuff_overlap(metadata):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])
    if len(thing_classes) == 0 or len(stuff_classes) == 0:
        return False

    if set(thing_classes).issubset(set(stuff_classes)) or set(stuff_classes).issubset(
        set(thing_classes)
    ):
        return True
    else:
        return False


def get_text_list(metadata, dataset_entity):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])

    if dataset_entity == "thing+stuff" and stuff_classes[0] == "things":
        text_list = list(thing_classes) + list(stuff_classes[1:])

    elif dataset_entity == "thing+stuff" and is_thing_stuff_overlap(metadata):
        text_list = thing_classes if len(thing_classes) > len(stuff_classes) else stuff_classes

    elif dataset_entity == "thing+stuff":
        text_list = list(thing_classes) + list(stuff_classes)

    elif dataset_entity == "stuff":
        text_list = list(stuff_classes)

    elif dataset_entity == "thing":
        text_list = list(thing_classes)

    return text_list


def get_stuff_score(box_cls, metadata, dataset_entity):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])

    semantic_box_cls = box_cls.clone()

    if dataset_entity == "thing+stuff" and stuff_classes[0] == "things":
        num_thing_classes = len(thing_classes)

        semantic_box_cls_0 = box_cls[..., :num_thing_classes]
        semantic_box_cls_1 = box_cls[..., num_thing_classes:]
        semantic_box_cls_0, _ = semantic_box_cls_0.min(dim=2, keepdim=True)
        semantic_box_cls = torch.cat([semantic_box_cls_0, semantic_box_cls_1], dim=2)

    if dataset_entity == "thing+stuff" and is_thing_stuff_overlap(metadata):
        semantic_box_cls = box_cls.clone()

    if dataset_entity == "stuff":
        semantic_box_cls = box_cls.clone()

    return semantic_box_cls
