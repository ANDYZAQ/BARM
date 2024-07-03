"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from tqdm import tqdm
import network
from network.modeling import get_modelmap
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.utils import AverageMeter, rand_bbox
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils.parser import get_argparser

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def visualization(images, labels, pred_labels, idx, root_path):
    """Visualize Segmentation Results and GT with opencv"""
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_labels = pred_labels#.cpu().numpy()
    
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        pred_label = pred_labels[i]
        
        image = image.transpose(1, 2, 0)
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)
        pred_label = pred_label.astype(np.uint8)

        # colorize segmentation map
        label_image = cv2.applyColorMap(label * 10, cv2.COLORMAP_JET)
        pred_label_image = cv2.applyColorMap(pred_label * 10, cv2.COLORMAP_JET)
        # set background label to black
        label_image[label == 0] = 0
        pred_label_image[pred_label == 0] = 0

        # overlay
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # concatenate with cv2.hconcat
        result_image = cv2.hconcat([image, label_image, pred_label_image])

        # save result image
        # print(f"save {root_path}/figs/{idx}_{i}.png")
        if not os.path.exists(f"{root_path}/figs/"):
            os.mkdir(f"{root_path}/figs/")
        cv2.imwrite(f"{root_path}/figs/{idx}_{i}.png", result_image)


def validate(opts, model, loader, device, metrics, root_path):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(tqdm(loader)):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs, _ = model(images)
            
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            # outputs[:, -50][outputs[:, -50] < 0.7] = 0
            pred_scores, preds = outputs.detach().max(dim=1) # .cpu().numpy()
            # preds[pred_scores < 0.7] = 0
            preds = preds.cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

            # visualization(images, labels, preds, i, root_path)
                
        score = metrics.get_results()
    return score


def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    # if opts.unknown: # [unknown, background, ...]
    #     opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    # fg_idx = 1 if opts.unknown else 0
    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:] 
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up model
    model_map = get_modelmap()

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=opts.bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)

    model = nn.DataParallel(model)
    model = model.to(device)
    
    dataset_dict = get_dataset(opts)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("... Testing Best Model")
    report_dict = dict()
    scheme = "overlap" if opts.overlap else "disjoint"
    root_path = f"checkpoints/{opts.subpath}/{opts.task}/{scheme}/step{opts.curr_step}/"
    ckpt_str = f"{root_path}%s_%s_%s_step_%d_{scheme}.pth"
    best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)

    checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    model.module.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_score = validate(opts=opts, model=model, loader=test_loader, 
                          device=device, metrics=metrics, root_path=root_path)
    print(metrics.to_str(test_score))
    report_dict[f'best/test_all_miou'] = test_score['Mean IoU']

    class_iou = list(test_score['Class IoU'].values())
    class_acc = list(test_score['Class Acc'].values())

    first_cls = len(get_tasks(opts.dataset, opts.task, 0)) 

    report_dict[f'best/test_before_mIoU'] = np.mean(class_iou[:first_cls]) 
    report_dict[f'best/test_after_mIoU'] = np.mean(class_iou[first_cls:])  
    report_dict[f'best/test_before_acc'] = np.mean(class_acc[:first_cls])  
    report_dict[f'best/test_after_acc'] = np.mean(class_acc[first_cls:])  

    print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
    print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':
            
    opts = get_argparser()
        
    total_step = len(get_tasks(opts.dataset, opts.task))
    # opts.curr_step = total_step - 1
    main(opts)

