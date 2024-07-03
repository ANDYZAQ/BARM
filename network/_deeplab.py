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


class DeepLabHeadCls(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadCls, self).__init__()
        self.add_channels = 32

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
        if len(num_classes) >= 2:
            self.ear = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(256, self.add_channels*c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.add_channels*c),
                    nn.ReLU(inplace=True),
                ) for c in num_classes[2:]])
        self.head = nn.ModuleList([
            nn.Conv2d(256+c*(idx)*self.add_channels, c+1, 1) for idx, c in enumerate(num_classes[1:])
        ])
        # self.head = nn.ModuleList([
        #     nn.Conv2d(256+c*(idx)*16, sum(num_classes[:idx+2]), 1) for idx, c in enumerate(num_classes[1:])
        # ])
        
        
        self._init_weight()

    def forward(self, feature):
        
        feature = self.aspp(feature['out'])
        
        fea_pre = self.head_pre(feature)
        heads = []
        for idx, h in enumerate(self.head):
            if idx == 0:
                heads.append(h(fea_pre))
                prev_feat = fea_pre
            else:
                ear_feat = self.ear[idx-1](fea_pre)
                prev_feat = torch.cat([prev_feat, ear_feat], dim=1)
                heads.append(
                    h(prev_feat)
                )

        bg_res = sum([hf[:,0] for hf in heads])
        heads = torch.cat([bg_res.unsqueeze(1)] + [hf[:,1:] for hf in heads], dim=1)
        # heads = torch.cat(heads, dim=1)
        # bs, _, h, w = heads[0].shape
        # heads = [torch.cat([hf, torch.zeros([bs, heads[-1].shape[1]-hf.shape[1], h, w]).to(hf.device)], dim=1) for hf in heads]
        # heads = sum(heads)
        
        return heads, feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadEASPP(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadEASPP, self).__init__()
        self.add_channels = 2

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
        if len(num_classes) >= 2:
            self.ear_aspp = nn.ModuleList(
                ASPP(in_channels, aspp_dilate, out_channels=self.add_channels*c) for c in num_classes[2:]
            )
            self.ear = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(256+c*(idx+1)*self.add_channels, 256+c*(idx+1)*self.add_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256+c*(idx+1)*self.add_channels),
                    nn.ReLU(inplace=True),
                ) for idx,c in enumerate(num_classes[2:])])
        self.head = nn.ModuleList([
            nn.Conv2d(256+c*(idx)*self.add_channels, c+1, 1) for idx, c in enumerate(num_classes[1:])
        ])
        # self.head = nn.ModuleList([
        #     nn.Conv2d(256+c*(idx)*16, sum(num_classes[:idx+2]), 1) for idx, c in enumerate(num_classes[1:])
        # ])
        
        
        self._init_weight()

    def forward(self, feature):
        
        fea_pre = self.aspp(feature['out'])
        
        fea_pre = self.head_pre(fea_pre)
        heads = []
        for idx, h in enumerate(self.head):
            if idx == 0:
                prev_feat = fea_pre
                heads.append(h(fea_pre))
            else:
                ear_feat = self.ear_aspp[idx-1](feature['out'])
                prev_feat = torch.cat([prev_feat, ear_feat], dim=1)
                ear_feat = self.ear[idx-1](prev_feat)
                heads.append(
                    h(ear_feat)
                )

        bg_res = sum([hf[:,0] for hf in heads])
        # if self.head[-1].weight.requires_grad is True:
        #     prev_bg = sum([hf[:,0] for hf in heads[:-1]])
        #     prev_head = torch.cat([prev_bg.unsqueeze(1)] + [hf[:,1:] for hf in heads[:-1]], dim=1)
        # else:
        #     prev_head = None
        heads = torch.cat([bg_res.unsqueeze(1)] + [hf[:,1:] for hf in heads], dim=1)
        # bs, _, h, w = heads[0].shape
        # heads = [torch.cat([hf, torch.zeros([bs, heads[-1].shape[1]-hf.shape[1], h, w]).to(hf.device)], dim=1) for hf in heads]
        # heads = sum(heads)
        
        return heads, feature['out'] #{'feature': prev_feat, 'novel_prev': prev_head} #

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadClsE(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadClsE, self).__init__()
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


class DeepLabHeadClsS(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadClsS, self).__init__()
        self.add_channels = 256

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
            
        self.head = nn.ModuleList([nn.Conv2d(256, num_classes[1]+1, 1)])
        for idx in range(2, len(num_classes)):
            c = num_classes[idx]
            self.head.append(nn.Conv2d(256, c+1, 1))

        self.curr_num_cls = num_classes[-1]
        
        self._init_weight()

    def forward(self, feature):
        
        feature_aspp = self.aspp(feature['out'])
        
        fea_pre = self.head_pre(feature_aspp)
        heads = [h(fea_pre) for h in self.head]
        prev_heads = heads

        if len(self.head) == 1:
            bg_res = heads[0][:,0:1]
        elif self.training:
            detached_bg_adapt = self.head[-1](fea_pre.detach())
            bg_res = torch.stack([heads[0][:,0].detach()]
                                 +[torch.clamp(hf[:,0], max=0).detach() for hf in heads[1:-1]]
                                # +[hf[:,0].detach() for hf in heads[1:-1]]
                                #  +[heads[-1][:,0]], dim=1).sum(dim=1, keepdim=True)
                                 + detached_bg_adapt[:, 0], dim=1).sum(dim=1, keepdim=True)
        else:
            bg_res = torch.stack([heads[0][:,0]]+[torch.clamp(hf[:,0], max=0) for hf in heads[1:]], dim=1).sum(dim=1, keepdim=True)
        heads = torch.cat([bg_res] + [hf[:,1:] for hf in heads], dim=1)
        
        return heads, {
            'feature': feature_aspp, 
            # 'prev_head': prev_head, 
            'back_out': feature['out'], 
            'fea_pre': fea_pre,
            'prev_heads': prev_heads,
        }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadClsSb(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadClsSb, self).__init__()
        self.add_channels = 32

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
            
        self.head = nn.ModuleList([nn.Conv2d(256, num_classes[1]+1, 1)])
        for idx in range(2, len(num_classes)):
            c = num_classes[idx]
            self.head.append(nn.Conv2d(self.add_channels, c+1, 1))
        
        self._init_weight()

    def forward(self, feature):
        
        feature_aspp = self.aspp(feature['out'])
        
        fea_pre = self.head_pre(feature_aspp)
        fea_inc = [e(feature_aspp) for e in self.ear]
        fea_inc = [fea_pre] + fea_inc
        # heads = [h(fea_pre) for h in self.head]
        heads = [h(feat) for h, feat in zip(self.head, fea_inc)]

        if self.head[-1].weight.requires_grad is True:
            prev_bg = sum([hf[:,0] for hf in heads[:-1]])
            prev_head = torch.cat([prev_bg.unsqueeze(1)] + [hf[:,1:] for hf in heads[:-1]], dim=1)
        else:
            prev_head = None

        # for idx in range(0, len(heads)-1):
        #     heads[idx] = heads[idx].detach() 
        # bg_res = sum([hf[:,0] for hf in heads])
        bg_res = sum([hf[:,0].detach() for hf in heads[:-1]] + [heads[-1][:,0]])
        heads = torch.cat([bg_res.unsqueeze(1)] + [hf[:,1:] for hf in heads], dim=1)
        # heads = torch.cat(heads, dim=1)
        # bs, _, h, w = heads[0].shape
        # heads = [torch.cat([hf, torch.zeros([bs, heads[-1].shape[1]-hf.shape[1], h, w]).to(hf.device)], dim=1) for hf in heads]
        # heads = sum(heads)
        
        return heads, {
            'feature': feature_aspp, 
            'prev_head': prev_head, 
            'back_out': feature['out']
        }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# class DeepLabHeadCls(nn.Module): # CLSVEC
#     def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
#         super(DeepLabHeadCls, self).__init__()

#         self.aspp = ASPP(in_channels, aspp_dilate)
        
#         self.head = nn.ModuleList(
#             [
#                  nn.Sequential(
#                     nn.Conv2d(256+sum(num_classes[:idx]) if idx > 1 else 256, 256, 3, padding=1, bias=False),
#                     nn.BatchNorm2d(256),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(256, c, 1)
#                 ) for idx, c in enumerate(num_classes)]
#         )
#         # self.adapt = nn.ModuleList()
#         # for idx, c in enumerate(num_classes[2:]):
#         #     for _ in range(num_classes[idx+2]):
#         #         self.adapt.append(nn.Sequential(
#         #             nn.Conv2d(sum(num_classes[:idx+2])+1, 2*sum(num_classes[:idx+2]), 1, bias=False),
#         #             nn.BatchNorm2d(2*sum(num_classes[:idx+2])),
#         #             nn.ReLU(inplace=True),
#         #             nn.Conv2d(2*sum(num_classes[:idx+2]), 1, 1)
#         #         ))
#         # self.new_classes = num_classes[2:]
        
#         self._init_weight()

#     def forward(self, feature):
        
#         feature = self.aspp(feature['out'])
        
#         # feature = self.head_pre(feature)
#         # heads = [h(feature) for h in self.head]
#         # heads_new = [hf.clone() for hf in heads]
#         # if len(heads_new) > 2:
#         #     heads_new[2:] = [torch.cat([self.adapt[idx*self.new_classes[idx]+cls](torch.cat(heads[:2]+[hf[:,cls:cls+1,...]], dim=1)) for cls in range(self.new_classes[idx])], dim=1) for idx, hf in enumerate(heads_new[2:])]
#             # heads_new[2:] = [torch.cat([a(torch.cat(heads[:2]+heads[idx+2][i:i+1]), dim=1) for i, a in enumerate(al)], dim=1) for idx, al in enumerate(self.adapt)]
#         heads = []
#         for h in self.head:
#             if len(heads)>1:
#                 heads.append(h(torch.cat([feature] + heads, dim=1)))
#             else:
#                 heads.append(h(feature))
#         heads = torch.cat(heads, dim=1)
#         # heads_new = torch.cat(heads_new, dim=1)
        
#         return heads, feature

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#     def _head_initialize(self):
#         for m in self.head:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)

class DeepLabHeadwob(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadwob, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head_pre = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),)
        self.head = nn.ModuleList([
            nn.Conv2d(256, c, 1) for c in num_classes[1:]
        ])
        self.pivot = 0 if num_classes[1] != 1 else 1
        
        self._init_weight()

    def forward(self, feature):
        
        feature = self.aspp(feature['out'])
        
        feature = self.head_pre(feature)
        heads = [h(feature) for h in self.head]
        heads = torch.cat(heads, dim=1)

        # insert generated background into pivot position
        bg = 1 - heads.sum(dim=1, keepdim=True)
        heads = torch.cat([heads[:, 0:self.pivot], bg, heads[:, self.pivot:]], dim=1)
        
        return heads, feature

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


class DeepLabMLP(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabMLP, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, c, 1)
            ) for c in num_classes
        ])
        
        self._init_weight()

    def forward(self, feature):
        
        feature = self.aspp(feature['out'])
        
        # feature = self.head_pre(feature)
        heads = [h(feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads, feature

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