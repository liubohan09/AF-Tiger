# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      Scale)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator,images_to_levels,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from .nanodet_head import Integral
from mmdet.core import distance2bbox,bbox2distance,bbox_overlaps
from mmdet.core.utils import filter_scores_and_topk

@HEADS.register_module()
class PicoHead(BaseDenseHead,BBoxTestMixin):
    """
    PicoHead
    Args:
        num_classes (int): Number of classes
        strides (list): The stride of each FPN Layer
        loss_cls (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """

    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 feat_channels=96,
                 kernel_size=3,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 stacked_convs=2,
                 use_l1=False,
                 share_cls_reg=True,
                 conv_cfg=None,
                 conv_bias='auto',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 loss_cls=dict(type='VarifocalLoss'),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_bbox=dict(type='GIoULoss',loss_weight=2),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=0.25),
                 reg_max=7,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01)
                 )
                 ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels=feat_channels
        self.kernel_size = kernel_size
        self.dcn_on_last_conv = dcn_on_last_conv
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max
        self.conv_bias = conv_bias
        self.use_depthwise = use_depthwise
        self.distribution_project = Integral(self.reg_max)
        self.share_cls_reg = share_cls_reg

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_dfl = build_loss(loss_dfl)
        self.loss_bbox = build_loss(loss_bbox)

        self.use_l1 = use_l1
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self._init_layers()
        # self.scales = nn.ModuleList(
        #     [Scale(1.0) for _ in self.prior_generator.strides])
    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.gfl_cls = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())  # conv layers of a single level head
            conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels + 4 * (self.reg_max + 1) if self.share_cls_reg else self.cls_out_channels, 1)
            self.gfl_cls.append(conv_cls)
            if not self.share_cls_reg:
                self.multi_level_reg_convs.append(self._build_stacked_convs())
                conv_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1)
                self.reg_convs.append(conv_reg)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=(self.kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def forward(self, feats):
        cls_logits_list = []
        bboxes_reg_list = []
        for i, fpn_feat in enumerate(feats):
            conv_cls_feat=self.multi_level_cls_convs[i](fpn_feat)
            cls_score = self.gfl_cls[i](conv_cls_feat)
            if self.share_cls_reg:
                cls_score,bbox_pred = cls_score.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            else:
                conv_reg_feat=self.multi_level_reg_convs[i](fpn_feat)
                bbox_pred =self.reg_convs[i](conv_reg_feat)

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)
        return (cls_logits_list, bboxes_reg_list)



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses.

        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]  # [48,24,12,6]
        # get grid cells of one image
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)
        center_priors=flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        center_priors=center_priors.reshape(num_imgs,-1,4)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
            for bbox_pred in bbox_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        dis_preds_feature = self.distribution_project(flatten_bbox_preds)
        dis_preds = dis_preds_feature * center_priors[..., 2, None]
        decoded_bboxes = self._bbox_decode(center_priors[..., :2], dis_preds)

        batch_assign_res = multi_apply(
            self._get_target_single,
            flatten_cls_preds.detach(),
            center_priors,
            decoded_bboxes.detach(),
            gt_bboxes,
            gt_labels,
        )

        loss_dict = self._get_loss_from_assign(
            flatten_cls_preds,flatten_bbox_preds,dis_preds_feature, decoded_bboxes, batch_assign_res
        )
        return loss_dict

    def _get_loss_from_assign(self, cls_preds, reg_preds, dis_preds_feature,decoded_bboxes, assign):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos),dtype=torch.float).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        # label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)



        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)


        vfl_score = torch.zeros_like(cls_preds)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            pos_labels = torch.gather(input=labels,index=pos_inds,dim=0)
            bbox_iou =bbox_overlaps(decoded_bboxes[pos_inds],bbox_targets[pos_inds],is_aligned=True)
            vfl_score[pos_inds, pos_labels] = bbox_iou

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
        loss_cls =self.loss_cls(cls_preds,vfl_score,avg_factor=num_total_samples)

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        if self.use_l1:
            if len(pos_inds) > 0:
                dis_preds=dis_preds_feature.reshape(-1, 4)
                loss_l1 =self.loss_l1(
                    dis_preds[pos_inds],
                    dist_targets[pos_inds],
                    weight=weight_targets[:, None].expand(-1, 4),
                )/bbox_avg_factor

                loss_dict.update(loss_l1=loss_l1)
        return  loss_dict

    @torch.no_grad()
    def _get_target_single(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """
        """
        num_priors = center_priors.size(0)
        device = center_priors.device
        gt_bboxes = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0
        # offset_priors = torch.cat(
        #     [center_priors[:, :2] + center_priors[:, 2:] * 0.5, center_priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        sampling_result = self.sampler.sample(assign_result, center_priors, gt_bboxes)

        pos_inds =sampling_result.pos_inds
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        pos_assigned_gt_inds =sampling_result.pos_assigned_gt_inds

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        center_priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        center_priors = center_priors.reshape(num_imgs, -1, 4)
        dis_preds = self.distribution_project(flatten_bbox_preds) * center_priors[..., 2, None]

        flatten_bboxes = self._bbox_decode(center_priors, dis_preds).to(torch.float32)

        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            bboxes = flatten_bboxes[img_id]

            result_list.append(
                self._bboxes_nms(cls_scores, bboxes,cfg))

        return result_list

    def _bboxes_nms(self, cls_scores, bboxes, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    def _bbox_decode(self, priors, bbox_preds,max_shape=None):
        return distance2bbox(priors, bbox_preds,max_shape)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors=None,
                    img_metas=None,
                    with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shape = img_metas[0]['img_shape_for_onnx']

        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        # e.g. Retina, FreeAnchor, etc.
        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
            mlvl_score_factor = [
                score_factors[i].detach() for i in range(num_levels)
            ]
            mlvl_score_factors = []

        mlvl_batch_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, score_factors, priors in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(0, 2, 3,
                                       1).reshape(batch_size, -1,
                                                  self.cls_out_channels)
            scores = scores.sigmoid()
            nms_pre_score = scores
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4*(self.reg_max+1))

            priors = priors.expand(batch_size, -1, priors.size(-1))
            bbox_pred = self.distribution_project(bbox_pred) * priors[..., 2, None]
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                nms_pre_score = nms_pre_score
            max_scores, _ = nms_pre_score.max(-1)
            _, topk_inds = max_scores.topk(nms_pre)
            batch_inds = torch.arange(
                batch_size, device=bbox_pred.device).view(
                -1, 1).expand_as(topk_inds).long()
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
            transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
            priors = priors.reshape(
                -1, priors.size(-1))[transformed_inds, :].reshape(
                batch_size, -1, priors.size(-1))
            bbox_pred = bbox_pred.reshape(-1,
                                          4)[transformed_inds, :].reshape(
                batch_size, -1, 4)
            scores = scores.reshape(
                -1, self.cls_out_channels)[transformed_inds, :].reshape(
                batch_size, -1, self.cls_out_channels)

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_scores, dim=1)
        from mmdet.core.export import add_dummy_nms_for_onnx
        if with_nms:
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        else:
            return batch_bboxes, batch_scores

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """Compute loss of the head.
    #
    #     """
    #     num_imgs = len(img_metas)
    #     num_level_anchors = [
    #         featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores
    #     ]#每个level的anchor数[48*48,....]
    #     featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]#[48,24,12,6]
    #     mlvl_priors = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=cls_scores[0].dtype,
    #         device=cls_scores[0].device,
    #         with_stride=True)#list，(coord_x, coord_y, stride_w, stride_h).
    #     # decode_bbox_preds = []
    #     # for bbox_pred ,mlvl_prior,stride in zip(bbox_preds, mlvl_priors, self.strides):
    #     #     center_in_feature = mlvl_prior.reshape(
    #     #         [-1, 4])[:, :-2] / stride
    #     #     bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
    #     #     pred_distances = self.integral(bbox_pred)  # 预测的box，[B,每层的anchor数，4],4代表距离anchor点的上下左右距离
    #     #     decode_bbox_pred_wo_stride = distance2bbox(center_in_feature,pred_distances).reshape([num_imgs,-1,4])
    #     #     decode_bbox_preds.append(decode_bbox_pred_wo_stride*stride)
    #
    #
    #     flatten_cls_preds = [
    #         cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
    #                                              self.cls_out_channels)
    #         for cls_pred in cls_scores
    #     ]#预测的cls [B,N,cls_n]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
    #         for bbox_pred in bbox_preds
    #     ]#预测的box，
    #     flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    #     flatten_priors = torch.cat(mlvl_priors)
    #
    #     flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)#distance_point_bbox_coder要实现这个
    #     pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []
    #     (pos_num_l, label_l,label_weight_l, bbox_targets) = multi_apply(
    #         self._get_target_single, flatten_cls_preds.detach(),
    #         flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
    #         flatten_bboxes.detach(), gt_bboxes, gt_labels)
    #
    #     labels = torch.tensor(np.stack(label_l, axis=0))
    #     label_weights = torch.tensor(np.stack(label_weight_l, axis=0))
    #     bbox_targets = torch.tensor(np.stack(bbox_target_l, axis=0))
    #     center_and_strides_list = images_to_levels(
    #         flatten_priors, num_level_anchors)
    #     labels_list = images_to_levels(labels,num_level_anchors)
    #     label_weights_list = images_to_levels(label_weights,
    #                                                 num_level_anchors)
    #     bbox_targets_list = images_to_levels(bbox_targets,
    #                                                num_level_anchors)
    #     num_pos = torch.tensor(
    #         sum(pos_num_l),
    #         dtype=torch.float,
    #         device=flatten_cls_preds.device)
    #     num_total_samples = max(reduce_mean(num_pos), 1.0)
    #
    #     loss_bbox_list, loss_dfl_list, loss_vfl_list, loss_l1_list, avg_factor = [], [], [], [], []
    #     for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
    #             cls_scores, bbox_preds, center_and_strides_list, labels_list,
    #             label_weights_list, bbox_targets_list, self.fpn_stride):
    #         center_and_strides = center_and_strides.reshape([-1, 4])
    #         cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
    #             [-1, self.cls_out_channels])
    #         bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
    #             [-1, 4 * (self.reg_max + 1)])  # [ ,4*9]
    #         bbox_targets = bbox_targets.reshape([-1, 4])
    #         labels = labels.reshape([-1])
    #
    #         bg_class_ind = self.num_classes
    #         pos_inds = torch.nonzero(
    #             torch.logical_and((labels >= 0), (labels < bg_class_ind)),
    #             as_tuple=False).squeeze(1)
    #         # vfl
    #         vfl_score = np.zeros(cls_score.shape)
    #
    #         if len(pos_inds) > 0:
    #             pos_bbox_targets = torch.gather(bbox_targets, pos_inds, axis=0)
    #             pos_bbox_pred = torch.gather(bbox_pred, pos_inds, axis=0)
    #             pos_centers = torch.gather(
    #                 center_and_strides[:, :-2], pos_inds, axis=0) / stride
    #
    #             weight_targets = F.sigmoid(cls_score.detach())
    #             weight_targets = torch.gather(
    #                 weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
    #             pos_bbox_pred_corners = self.integral(pos_bbox_pred)  # [,4]
    #             pos_decode_bbox_pred = distance2bbox(pos_centers,
    #                                                  pos_bbox_pred_corners)
    #             pos_decode_bbox_targets = pos_bbox_targets / stride
    #             bbox_iou = bbox_overlaps(
    #                 pos_decode_bbox_pred.detach().numpy(),
    #                 pos_decode_bbox_targets.detach().numpy(),
    #                 is_aligned=True)
    #
    #             # vfl
    #             pos_labels = torch.gather(labels, pos_inds, axis=0)
    #             vfl_score[pos_inds.numpy(), pos_labels] = bbox_iou
    #
    #             pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
    #             target_corners = bbox2distance(pos_centers,
    #                                            pos_decode_bbox_targets,
    #                                            self.reg_max).reshape([-1])
    #             if self.use_l1:
    #                 loss_l1 = (torch.abs(pos_decode_bbox_pred - pos_decode_bbox_targets) * weight_targets).sum(
    #                 )
    #                 loss_l1_list.append(loss_l1)
    #                 # regression loss
    #             loss_bbox = torch.sum(
    #                 self.loss_bbox(pos_decode_bbox_pred,
    #                                pos_decode_bbox_targets) * weight_targets)
    #
    #             # dfl loss
    #             loss_dfl = self.loss_dfl(
    #                 pred_corners,
    #                 target_corners,
    #                 weight=weight_targets.expand([-1, 4]).reshape([-1]),
    #                 avg_factor=4.0)
    #         else:
    #             loss_bbox = bbox_pred.sum() * 0
    #             loss_dfl = bbox_pred.sum() * 0
    #             weight_targets = torch.tensor([0], dtype='float32')
    #
    #         # vfl loss
    #         num_pos_avg_per_gpu = num_total_samples
    #         vfl_score = torch.tensor(vfl_score)
    #         loss_vfl = self.loss_vfl(
    #             cls_score, vfl_score, avg_factor=num_pos_avg_per_gpu)
    #
    #         loss_bbox_list.append(loss_bbox)
    #         loss_dfl_list.append(loss_dfl)
    #         loss_vfl_list.append(loss_vfl)
    #
    #         avg_factor.append(weight_targets.sum())
    #     avg_factor = torch.tensor(
    #         sum(avg_factor),
    #         dtype=torch.float,
    #         device=flatten_cls_preds.device)
    #     avg_factor = max(reduce_mean(avg_factor), 1.0)
    #
    #     if avg_factor <= 0:
    #         loss_vfl = torch.tensor(0, dtype='float32', requires_grad=False)
    #         loss_bbox = torch.tensor(
    #             0, dtype='float32', requires_grad=False)
    #         loss_dfl = torch.tensor(0, dtype='float32', requires_grad=False)
    #     else:
    #         losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
    #         losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
    #         losses_l1 = list(map(lambda x: x / avg_factor, loss_l1_list))
    #         loss_vfl = sum(loss_vfl_list)
    #         loss_bbox = sum(losses_bbox)
    #         loss_dfl = sum(losses_dfl)
    #         loss_l1 = sum(losses_l1)
    #         loss_bbox = loss_bbox + loss_l1
    #
    # def _get_target_single(self, flatten_cls_pred, flatten_center_and_stride,
    #                        flatten_bbox, gt_bboxes, gt_labels):
    #     """Compute targets for priors in a single image.
    #     """
    #     num_bboxes = flatten_bbox.shape[0]
    #     assign_result = self.assigner.assign(
    #         flatten_cls_pred.sigmoid(), flatten_center_and_stride,
    #         flatten_bbox, gt_bboxes, gt_labels)
    #     sampling_result =self.sampler.sample(assign_result,flatten_center_and_stride,gt_bboxes)
    #     pos_inds =sampling_result.pos_inds
    #     neg_inds =sampling_result.neg_inds
    #     pos_gt_bboxes =sampling_result.pos_gt_bboxes
    #     pos_assigned_gt_inds =sampling_result.pos_assigned_gt_inds
    #     bbox_target = np.zeros_like(flatten_bbox)
    #     bbox_weight = np.zeros_like(flatten_bbox)
    #     label = np.ones([num_bboxes], dtype=np.int64) * self.num_classes
    #     label_weight = np.zeros([num_bboxes], dtype=np.float32)
    #
    #     if len(pos_inds) > 0:
    #         gt_labels = gt_labels.numpy()
    #         pos_bbox_targets = pos_gt_bboxes
    #         bbox_target[pos_inds, :] = pos_bbox_targets
    #         bbox_weight[pos_inds, :] = 1.0
    #         if not np.any(gt_labels):
    #             label[pos_inds] = 0
    #         else:
    #             label[pos_inds] = gt_labels.squeeze(-1)[pos_assigned_gt_inds]
    #
    #         label_weight[pos_inds] = 1.0
    #     if len(neg_inds) > 0:
    #         label_weight[neg_inds] = 1.0
    #
    #     pos_num = max(pos_inds.size, 1)
    #     return pos_num, label, label_weight, bbox_target
    #
    # def _bbox_decode(self, priors, bbox_preds):
    #     tl_x = priors[:, 0] - bbox_preds[:, 0] * priors[:, -1]
    #     tl_y = priors[:, 1] - bbox_preds[:, 1] * priors[:, -1]
    #     br_x = priors[:, 0] + bbox_preds[:, 0] * priors[:, -1]
    #     br_y = priors[:, 0] + bbox_preds[:, 1] * priors[:, -1]
    #
    #     decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    #     return decoded_bboxes
    #
    # def _get_bboxes_single(self,
    #                        cls_score_list,
    #                        bbox_pred_list,
    #                        score_factor_list,
    #                        mlvl_priors,
    #                        img_meta,
    #                        cfg,
    #                        rescale=False,
    #                        with_nms=True,
    #                        **kwargs):
    #     """Transform outputs of a single image into bbox predictions.
    #
    #     Args:
    #         cls_score_list (list[Tensor]): Box scores from all scale
    #             levels of a single image, each item has shape
    #             (num_priors * num_classes, H, W).
    #         bbox_pred_list (list[Tensor]): Box energies / deltas from
    #             all scale levels of a single image, each item has shape
    #             (num_priors * 4, H, W).
    #         score_factor_list (list[Tensor]): Score factor from all scale
    #             levels of a single image. GFL head does not need this value.
    #         mlvl_priors (list[Tensor]): Each element in the list is
    #             the priors of a single level in feature pyramid, has shape
    #             (num_priors, 4).
    #         img_meta (dict): Image meta info.
    #         cfg (mmcv.Config): Test / postprocessing configuration,
    #             if None, test_cfg would be used.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    #
    #     Returns:
    #         tuple[Tensor]: Results of detected bboxes and labels. If with_nms
    #             is False and mlvl_score_factor is None, return mlvl_bboxes and
    #             mlvl_scores, else return mlvl_bboxes, mlvl_scores and
    #             mlvl_score_factor. Usually with_nms is False is used for aug
    #             test. If with_nms is True, then return the following format
    #
    #             - det_bboxes (Tensor): Predicted bboxes with shape \
    #                 [num_bboxes, 5], where the first 4 columns are bounding \
    #                 box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
    #                 column are scores between 0 and 1.
    #             - det_labels (Tensor): Predicted labels of the corresponding \
    #                 box with shape [num_bboxes].
    #     """
    #     cfg = self.test_cfg if cfg is None else cfg
    #     img_shape = img_meta['img_shape']
    #     nms_pre = cfg.get('nms_pre', -1)
    #
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     mlvl_labels = []
    #     for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
    #             zip(cls_score_list, bbox_pred_list,
    #                 self.prior_generator.strides, mlvl_priors)):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         assert stride[0] == stride[1]
    #
    #         bbox_pred = bbox_pred.permute(1, 2, 0)
    #         bbox_pred = self.integral(bbox_pred) * stride[0]
    #
    #         scores = cls_score.permute(1, 2, 0).reshape(
    #             -1, self.cls_out_channels).sigmoid()
    #
    #         # After https://github.com/open-mmlab/mmdetection/pull/6268/,
    #         # this operation keeps fewer bboxes under the same `nms_pre`.
    #         # There is no difference in performance for most models. If you
    #         # find a slight drop in performance, you can set a larger
    #         # `nms_pre` than before.
    #         results = filter_scores_and_topk(
    #             scores, cfg.score_thr, nms_pre,
    #             dict(bbox_pred=bbox_pred, priors=priors))
    #         scores, labels, _, filtered_results = results
    #
    #         bbox_pred = filtered_results['bbox_pred']
    #         priors = filtered_results['priors']
    #
    #         bboxes = self.bbox_coder.decode(
    #             self.anchor_center(priors), bbox_pred, max_shape=img_shape)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #         mlvl_labels.append(labels)
    #
    #     return self._bbox_post_process(
    #         mlvl_scores,
    #         mlvl_labels,
    #         mlvl_bboxes,
    #         img_meta['scale_factor'],
    #         cfg,
    #         rescale=rescale,
    #         with_nms=with_nms)
