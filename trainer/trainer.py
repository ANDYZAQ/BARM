from tqdm import tqdm
from datasets import *
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.ckpt import save_ckpt
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils import *
from network import get_modelmap

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import json


class Trainer(object):
    def __init__(self, opts, device) -> None:
        super(Trainer, self).__init__()
        self.opts = opts
        self.device = device
        self.model_name = opts.model
        self.num_classes = opts.num_classes
        self.output_stride = opts.output_stride
        self.bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
        self.separable_conv = opts.separable_conv
        self.curr_step = opts.curr_step
        self.lr = opts.lr
        self.weight_decay = opts.weight_decay
        self.overlap = opts.overlap
        self.dataset = opts.dataset
        self.task = opts.task
        self.pseudo = opts.pseudo
        self.pseudo_thresh = opts.pseudo_thresh
        self.loss_type = opts.loss_type
        self.amp = opts.amp
        self.batch_size = opts.batch_size
        self.ckpt = opts.ckpt
        self.train_epoch = opts.train_epoch
        self.local_rank = opts.local_rank

        self.curr_idx = [
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
        ]
        
        self.init_models()
        # Set up metrics
        self.metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)

        scheme = "overlap" if opts.overlap else "disjoint"
        root_path = f"checkpoints/{opts.subpath}/{self.task}/{scheme}/step{opts.curr_step}/"
        if self.local_rank == 0:
            utils.mkdir(root_path)
        root_path_prev = f"checkpoints/{opts.subpath}/{self.task}/{scheme}/step{opts.curr_step-1}/"
        self.ckpt_str = f"{root_path}%s_%s_%s_step_%d_{scheme}.pth"
        self.ckpt_str_prev = f"{root_path_prev}%s_%s_%s_step_%d_{scheme}.pth"
        self.root_path = root_path
        self.root_path_prev = root_path_prev
        self.init_ckpt()
        self.train_loader, self.val_loader, self.test_loader, self.memory_loader = init_dataloader(opts)
        
        self.init_iters(opts)
        self.scheduler = build_scheduler(opts, self.optimizer, self.total_itrs)
        self.criterion = build_criterion(opts)
        self.avg_loss = AverageMeter()
        self.avg_time = AverageMeter()
        self.avg_loss_std = AverageMeter()
        self.aux_loss_1 = AverageMeter()
        self.aux_loss_2 = AverageMeter()
        self.aux_loss_3 = AverageMeter()
        self.aux_loss_4 = AverageMeter()
        self.logger = Logger(root_path)

        # self.kl_loss = nn.KLDivLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def init_models(self):
        # Set up model
        model_map = get_modelmap()

        if self.local_rank==0:
            print(f"Category components: {self.num_classes}")
        self.model = model_map[self.model_name](num_classes=self.num_classes, output_stride=self.output_stride, bn_freeze=self.bn_freeze)
        if self.separable_conv and 'plus' in self.model_name:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        if torch.cuda.device_count() > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.bn_freeze:
                self.model.freeze_bn()
                self.model.freeze_dropout()
            
        if self.curr_step > 0:
            """ load previous model """
            self.model_prev = model_map[self.model_name](num_classes=self.num_classes[:-1], output_stride=self.output_stride, bn_freeze=self.bn_freeze)
            if self.separable_conv and 'plus' in self.model_name:
                network.convert_to_separable_conv(self.model_prev.classifier)
            utils.set_bn_momentum(self.model_prev.backbone, momentum=0.01)
            self.model_prev = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_prev)
            self.model_prev.freeze_bn()
            self.model_prev.freeze_dropout()
        else:
            self.model_prev = None

        self.optimizer = self.init_optimizer()

        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        else:
            self.model = nn.DataParallel(self.model)
        self.model.train()
        
        if self.curr_step > 0:
            self.model_prev = self.model_prev.to(self.device)
            self.model_prev = nn.parallel.DistributedDataParallel(self.model_prev, find_unused_parameters=True)
            self.model_prev.eval()
            for param in self.model_prev.module.parameters():
                param.requires_grad = False
        
    def init_optimizer(self):
        # Set up optimizer & parameters
        if self.curr_step > 0:
            lower_lr = 1e-4
            training_params = [{'params': self.model.classifier.head[-1].parameters(), 'lr': self.lr}, ]
            training_params.append({'params': self.model.classifier.ear[-1].parameters(), 'lr': self.lr})

            training_params.append({'params': self.model.classifier.aspp.parameters(), 'lr': 1e-3})
            training_params.append({'params': self.model.classifier.head[0:-1].parameters(), 'lr': lower_lr})
            training_params.append({'params': self.model.classifier.ear[0:-1].parameters(), 'lr': lower_lr})
            training_params.append({'params': self.model.classifier.head_pre.parameters(), 'lr': lower_lr})
            training_params.append({'params': self.model.backbone.parameters(), 'lr': lower_lr})
        else:
            training_params = [{'params': self.model.backbone.parameters(), 'lr': 0.001},
                            {'params': self.model.classifier.parameters(), 'lr': 0.01}]
            
        optimizer = torch.optim.SGD(params=training_params, 
                                    lr=self.lr, 
                                    momentum=0.9, 
                                    weight_decay=self.weight_decay,
                                    nesterov=True)
        
        if self.local_rank == 0:
            print("----------- trainable parameters --------------")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
            print("-----------------------------------------------")

        return optimizer
    
    def init_ckpt(self):
        if self.curr_step > 0: # previous step checkpoint
            self.ckpt = self.ckpt_str_prev % (self.model_name, self.dataset, self.task, self.curr_step-1)
        else:
            return
        if self.curr_step > 1:
            self.ckpt = self.root_path_prev + "final.pth"

        print(self.ckpt)
        assert os.path.isfile(self.ckpt)
        checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["model_state"]
        self.model_prev.module.load_state_dict(checkpoint, strict=True)

        curr_head_num = len(self.model.module.classifier.head) - 1

        checkpoint[f"classifier.ear.{curr_head_num-1}.0.weight"] = checkpoint["classifier.head_pre.0.weight"]
        checkpoint[f"classifier.ear.{curr_head_num-1}.1.weight"] = checkpoint["classifier.head_pre.1.weight"]
        checkpoint[f"classifier.ear.{curr_head_num-1}.1.bias"] = checkpoint["classifier.head_pre.1.bias"]
        self.model.module.load_state_dict(checkpoint, strict=False)

        print("Model restored from %s" % self.ckpt)
        del checkpoint  # free memory
    
    def init_iters(self, opts):
         # Restore
        self.best_score = -1
        
        self.total_itrs = self.train_epoch * len(self.train_loader)
        self.val_interval = max(100, self.total_itrs // 100)
        print(f"... train epoch : {self.train_epoch} , iterations : {self.total_itrs} , val_interval : {self.val_interval}")
                
    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        # save_ckpt(self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step), self.model, self.optimizer, self.best_score)

        # =====  Train  =====
        for epoch in range(self.train_epoch):
            self.model.module.train()
            for seq, (images, labels, _) in enumerate(self.train_loader):
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs, loss = self.train_episode(images, labels, scaler, epoch)
                
                if self.local_rank==0 and seq % 10 == 0:
                    print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%4f, StdLoss=%.4f, A1=%.4f, A2=%.4f, A3=%.4f, A4=%.4f Time=%.2f , LR=%.8f" %
                        (self.task, self.curr_step, epoch, seq, len(self.train_loader), 
                        self.avg_loss.avg, self.avg_loss_std.avg, self.aux_loss_1.avg, self.aux_loss_2.avg, self.aux_loss_3.avg, self.aux_loss_4.avg, self.avg_time.avg*1000, self.optimizer.param_groups[0]['lr']))
                    self.logger.write_loss(self.avg_loss.avg, epoch * len(self.train_loader) + seq + 1)

            if self.local_rank == 0 and (len(self.train_loader) > 100 or epoch % 5 ==4):
                print("[Validation]")
                val_score = self.validate()
                print(self.metrics.to_str_val(val_score))
                
                class_iou = list(val_score['Class IoU'].values())
                val_score = np.mean( class_iou[self.curr_idx[0]:self.curr_idx[1]] + [class_iou[0]])
                curr_score = np.mean( class_iou[self.curr_idx[0]:self.curr_idx[1]] )
                print("curr_val_score : %.4f\n" % (curr_score))
                self.logger.write_score(curr_score, epoch)
                
                if curr_score > self.best_score:  # save best model
                    print("... save best ckpt : ", curr_score)
                    self.best_score = curr_score
                    save_ckpt(self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step), self.model, self.optimizer, self.best_score)
        
        if self.local_rank == 0 :
            save_ckpt(self.root_path+"final.pth", self.model, self.optimizer, self.best_score)
            print("... Training Done")
            if self.curr_step > 0:
                self.do_evaluate(mode='test')
        if torch.cuda.device_count() > 1:
            torch.distributed.barrier()


    def train_episode(self, images, labels, scaler, epoch):
        self.optimizer.zero_grad()
        end_time = time.time()
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=self.amp):

            outputs, features = self.model.module(images)

            if self.pseudo and self.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs_prev, features_prev = self.model_prev.module(images)
                outputs_prev = F.interpolate(outputs_prev, labels.shape[-2:], mode='bilinear', align_corners=False)
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)

                if self.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels= torch.where(
                    (labels==0) & (pred_labels>0) & (pred_scores >= self.pseudo_thresh), 
                    pred_labels, 
                    labels)
                # select pseudo mask on 0, 255 of labels
                positive_masks = torch.logical_and(labels!=0, labels!=255).float()
                negative_masks = (labels==0).float()

                adp_outputs = torch.cat([
                    outputs[:, 0:1], 
                    outputs[:, 1:-self.num_classes[-1]].detach(),
                    outputs[:, -self.num_classes[-1]:]
                ], dim=1)
                std_loss = self.criterion(adp_outputs, pseudo_labels)
                loss = std_loss.clone()
                
                co_bg, co_tn, co_p, co_oc = 5, 4, 1, 1

                feature, feature_prev = features["feature"], features_prev["feature"]
                pheads, pheads_prev = features["prev_heads"], features_prev["prev_heads"]
                back_out, back_out_prev = features["back_out"], features_prev["back_out"]
                ear_feats, ear_feats_prev = features["ear_feats"], features_prev["ear_feats"]
                
                positive_masks = F.interpolate(positive_masks.unsqueeze(1), feature.shape[-2:])
                negative_masks = F.interpolate(negative_masks.unsqueeze(1), feature.shape[-2:])

                # bga losses: bga_ce, bga_t
                bg_adp_bce = self.bce_loss(pheads[-1][:, 0:1], torch.zeros_like(pheads[-1][:, 0:1]))
                bga_ce_loss = (bg_adp_bce * positive_masks).sum() / (positive_masks.sum()+1e-8)
                mse_bg_0 = F.mse_loss(pheads[-1][:, 0:1].sigmoid(), torch.zeros_like(pheads[-1][:, 0:1]), reduction="none") * negative_masks
                mse_bg_1 = F.mse_loss(pheads[-1][:, 0:1].sigmoid(), torch.ones_like(pheads[-1][:, 0:1]), reduction="none") * negative_masks
                tri_bg_loss = torch.clamp(mse_bg_1 - mse_bg_0, min=0).sum() / negative_masks.sum()

                # ofd loss
                tunnel_loss = torch.zeros_like(loss)
                for i in range(len(ear_feats_prev)):
                    tunnel_loss += self.mse_loss(ear_feats[i]*negative_masks, ear_feats_prev[i]*negative_masks)

                # gkd_loss
                gkd_loss = torch.zeros_like(loss)
                for i in range(len(pheads_prev)):
                    phead_prev_sig = pheads_prev[i].sigmoid()
                    gkd_loss += self.bce_loss(pheads[i], phead_prev_sig).mean(dim=[0,2,3]).sum()
                loss += (co_bg * tri_bg_loss + co_tn * tunnel_loss + co_p * gkd_loss + co_oc*bga_ce_loss)

                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
                self.aux_loss_1.update(tri_bg_loss.item())
                self.aux_loss_2.update(tunnel_loss.item())
                self.aux_loss_3.update(gkd_loss.item())
                self.aux_loss_4.update(bga_ce_loss.item())
            else:
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
                loss = std_loss = self.criterion(outputs, labels)
            

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        self.scheduler.step()
        self.avg_loss.update(loss.item())
        self.avg_time.update(time.time() - end_time)
        self.avg_loss_std.update(std_loss.item())

        return outputs, loss
    
    def do_evaluate(self, mode='val'):
        print("[Testing Best Model]")
        # best_ckpt = self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step)
        best_ckpt = self.root_path+"final.pth"
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        self.model.module.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.module.eval()
        
        test_score = self.validate(mode)
        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.dataset, self.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))

        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mIoU'] = np.mean(class_iou[first_cls:])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mAcc'] = np.mean(class_acc[first_cls:])

        # save results as json
        with open(f"{self.root_path}/test_results.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()


    def validate(self, mode='val'):
        """Do validation and return specified samples"""
        self.metrics.reset()
        ret_samples = []
        self.model.module.eval()

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs, _ = self.model.module(images)
                
                if self.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)

                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.metrics.update(targets, preds)
                    
            score = self.metrics.get_results()
        return score