"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from datetime import timedelta
from tqdm import tqdm
import network

# from trainer.trainer2 import Trainer
from trainer.trainer import Trainer
# from trainer.trainer_s import Trainer
# from trainer.trainer_bg import Trainer # 不推测背景
# from trainer.trainer_fb import Trainer # 冻结backbone
# from trainer.trainer_pro import Trainer # 使用Prototype和MLP
# from trainer.trainer_cor import Trainer # 使用Correlation

import utils
import os
import time
import random
import numpy as np
import cv2


import torch
import torch.nn as nn
from utils.parser import get_argparser
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
import torch.distributed as dist

torch.backends.cudnn.benchmark = True


def main(opts, device):
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        
    # opts.target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    opts.target_cls = [get_tasks(opts.dataset, opts.task, step) for step in range(opts.curr_step+1)]
    
    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]
    
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
    
    trainer = Trainer(opts, device)

    if opts.test_only:
        trainer.do_evaluate(mode='test')
        return
        
    trainer.train()
    # trainer.prototype_for_train()

if __name__ == '__main__':
            
    opts = get_argparser()
    if torch.cuda.device_count() > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120))
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        opts.local_rank = rank
        os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
    else:
        opts.local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset, opts.task))
    
    if opts.initial:
        opts.curr_step = 0
        main(opts, device)
    else:
        for step in range(start_step, total_step):
            opts.curr_step = step
            main(opts, device)
    # main(opts)
    
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()