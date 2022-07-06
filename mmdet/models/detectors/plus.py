import copy
import torch.nn as nn
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from mmcv.runner import load_checkpoint

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class Plus(SingleStageDetector):
    """Implementation of ` <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_neck,
                 teacher_bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Plus,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)
        self.teacher_neck = build_neck(teacher_neck)
        self.teacher_neck.init_weights()
        self.teacher_head = build_head(teacher_bbox_head)
        self.teacher_head.init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        feat =self.backbone(img)
        fpn_feat=self.neck(feat)

        aux_fpn_feat = self.teacher_neck(feat)
        dual_fpn_feat = (
            torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
        )

        teacher_out =self.teacher_head(dual_fpn_feat)

        losses= self.bbox_head.forward_train(fpn_feat,img_metas, gt_bboxes, gt_labels,
                                        gt_bboxes_ignore,teacher_out)

        return losses
