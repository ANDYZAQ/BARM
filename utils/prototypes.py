import torch
import torch.nn as nn
import torch.nn.functional as F


def save_protos(features, labels):
    """Save prototypes from features"""
    prototypes = []
    categories = []
    # convert labels to one-hot
    labels = F.one_hot(labels, num_classes=labels.max()+1).permute(0, 3, 1, 2).float() # b, c, h, w
    labels = F.interpolate(labels, size=features.shape[2:], mode='nearest')
    features = features.detach()
    for cls in range(1, labels.shape[1]):
        label = labels[:, cls].unsqueeze(1) # b, 1, h, w
        for bs in range(features.shape[0]):
            lbsum = label[bs].sum()
            if lbsum == 0:
                continue
            feature = features[bs] * label[bs]
            pro = feature.sum(dim=(-2, -1)) / (lbsum + 1e-7) # b, c
            prototypes.append(pro.unsqueeze(-1).unsqueeze(-1))
            categories.append(cls)
    
    return prototypes, categories