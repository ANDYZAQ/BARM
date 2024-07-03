import copy
import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(304, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in num_classes]
        )
        
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        
        heads = [h(output_feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.normal_(m.weight, mean=5, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
        self.head = nn.ModuleList([
            nn.Conv2d(256, c, 1) for c in num_classes
        ])
        # self.head = nn.ModuleList(
        #     [
        #          nn.Sequential(
        #             nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(256, c, 1)
        #         ) for c in num_classes]
        # )
        
        self._init_weight()

    def forward(self, feature):
        back_out = feature['out']
        feature = self.aspp(back_out)
        
        feature = self.head_pre(feature)
        heads = [h(feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads, {
            "feature" : feature, 
            "back_out" : back_out
            }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class DeepLabHeadBgA(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadBgA, self).__init__()
        self.add_channels = 256

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
        if len(num_classes) >= 2:
            self.ear = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(256, self.add_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.add_channels),
                    nn.ReLU(inplace=True),
                ) for c in num_classes[2:]])

        # Additional channel for Background Adaptation
        self.head = nn.ModuleList([nn.Conv2d(256, num_classes[1]+1, 1)])
        for idx in range(2, len(num_classes)):
            c = num_classes[idx]
            self.head.append(
                nn.Conv2d(self.add_channels, c+1, 1)
            )
        
        self._init_weight()

    def forward(self, feature):
        back_out = feature['out']
        feature = self.aspp(feature['out'])
        
        fea_pre = self.head_pre(feature)
        heads = [self.head[0](fea_pre)]
        ear_feats = [fea_pre]
        for idx, h in enumerate(self.head[1:]):
            ear_feat = self.ear[idx](feature)
            ear_feats.append(ear_feat)
            heads.append(h(ear_feat))
        
        prev_heads = heads

        # Unification of Background Adaptations
        if len(self.head) == 1:
            # bg_res = heads[0][:,0:1]
            heads = torch.cat(heads, dim=1)
        elif self.training:
            bg_res = torch.stack([heads[0][:,0].detach()]
                                 +[torch.clamp(hf[:,0], max=0).detach() for hf in heads[1:-1]]
                                # +[hf[:,0].detach() for hf in heads[1:-1]]
                                 +[heads[-1][:,0]], dim=1).sum(dim=1, keepdim=True)
            heads = torch.cat([bg_res] + [hf[:,1:] for hf in heads], dim=1)
        else:
            bg_res = torch.stack([heads[0][:,0]]+[torch.clamp(hf[:,0], max=0) for hf in heads[1:]], dim=1).sum(dim=1, keepdim=True)
            heads = torch.cat([bg_res] + [hf[:,1:] for hf in heads], dim=1)
        
        return heads, {
            'feature': feature, 
            'prev_heads': prev_heads, 
            'ear_feats': ear_feats,
            'back_out': back_out,
        }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module