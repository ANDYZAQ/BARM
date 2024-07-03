from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation
from utils import ext_transforms as et
from torch.utils.data.distributed import DistributedSampler
import torch


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


def init_dataloader(opts):
    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)

    train_sampler = DistributedSampler(dataset_dict['train']) if torch.cuda.device_count() > 1 else None
    train_loader = data.DataLoader(
        dataset_dict['train'], shuffle=(train_sampler is None), sampler=train_sampler, batch_size=opts.batch_size, num_workers=4, pin_memory=True, drop_last=True)
    if opts.local_rank == 0:
        val_loader = data.DataLoader(
            dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = data.DataLoader(
            dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader, test_loader = None, None
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
        (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_sampler = DistributedSampler(dataset_dict['memory'])
        memory_loader = data.DataLoader(
            memory_sampler, batch_size=opts.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    else:
        memory_loader = None
    
    return train_loader, val_loader, test_loader, memory_loader