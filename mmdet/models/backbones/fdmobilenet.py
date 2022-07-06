import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from mmdet.models.builder import BACKBONES
from pytorchcv.model_provider import get_model as ptcv_get_model

@BACKBONES.register_module()
class FDMobilenet(nn.Module):
    def __init__(
            self,
            model_name,
            out_stages=(1, 2, 3),
            pretrained=True,
            **kwargs):
        super(FDMobilenet, self).__init__()
        self.out_stages=out_stages
        net=ptcv_get_model(model_name, pretrained=pretrained,**kwargs).features
        self.backbone = nn.ModuleList([
            nn.Sequential(net.init_block, net.stage1),
            net.stage2,
            net.stage3,
            net.stage4,
            # net.stage5
        ])

    def forward(self, x):
        outs = []
        x = self.backbone[0](x)
        for i in range(1, 4):
            x = self.backbone[i](x)
            if i in self.out_stages:
                outs.append(x)
        return tuple(outs)