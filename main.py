# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import datetime
import os
import random
import time
from argparse import Namespace
from logging import Logger
from pathlib import Path

import numpy as np
import sacred
import torch
import torch.multiprocessing
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                              SequentialSampler)

import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from experiment import new_experiment
from models import build_model, build_optimizer
from util.tool import load_model

torch.multiprocessing.set_sharing_strategy('file_system')

ex = new_experiment('RadarMOTR', training=True)
ex.add_config(os.path.join('configs', 'train.yaml'))
ex.add_named_config('debug', os.path.join('configs', 'train_debug.yaml'))
ex.add_named_config('resnet50', os.path.join('configs', 'resnet50.yaml'))
ex.add_named_config('resnet18', os.path.join('configs', 'resnet18.yaml'))
ex.add_named_config('motrv2', os.path.join('configs', 'motrv2.yaml'))


@ex.capture
def train(args: Namespace, _run, _log: Logger):
    utils.init_distributed_mode(args)

    if utils.is_main_process():
        sacred.commands.print_config(_run)
        _log.info(f'git {utils.get_sha()}')

    if args.frozen_weights is not None:
        assert args.masks, 'Frozen training is meant for segmentation only'

    assert args.batch_size == 1, 'Currently only a batch size of 1 is supported'
    assert torch.cuda.is_available(), 'CUDA is not available'

    # fix the seed for reproducibility
    set_seed(args.seed)

    # Set TF32 options
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32_matmul

    _log.info(f'Using backbone {args.backbone} with LR {args.lr_backbone}')

    model, criterion = build_model(args)
    device = torch.device(args.device)
    model.to(device, non_blocking=True)
    criterion.to(device, non_blocking=True)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info(f'Number of params: {n_parameters:,}')

    dataset_train = build_dataset(args.dataset, split='train', args=args)
    dataset_val = build_dataset(args.dataset, split='val', args=args)

    _log.info(f'Training on {args.dataset} dataset found in \"{args.data_path}\".')
    _log.info(f'Training Dataset: Found {len(dataset_train.sequences)} radar sequences with total of {len(dataset_train.frame_indices)} frames.')
    _log.info(f'Validation Dataset: Found {len(dataset_val.sequences)} radar sequences with total of {len(dataset_val.frame_indices)} frames.')

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train, shuffle=True)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

    batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = BatchSampler(sampler_val, args.batch_size, drop_last=False)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.mot_collate_fn,
                                   num_workers=args.workers,
                                   pin_memory=True)

    data_loader_val = DataLoader(dataset_val,
                                 batch_sampler=batch_sampler_val,
                                 collate_fn=utils.mot_collate_fn,
                                 num_workers=args.workers,
                                 pin_memory=True)

    optimizer = build_optimizer(model_without_ddp, args)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    learn_rates = lr_scheduler.get_last_lr()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if not args.eval and args.resume:
        model_without_ddp, optimizer, \
            lr_scheduler, start_epoch = load_model(model_without_ddp,
                                                   args.resume, resume=True,
                                                   optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                   lr_drop=args.lr_drop)
        args.start_epoch = start_epoch
    elif args.pretrained:
        model_without_ddp = load_model(model_without_ddp, args.pretrained, resume=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Start training.")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)

        # TRAINING
        train_stats = train_one_epoch(epoch, model, criterion, data_loader_train, optimizer, device, args)
        log_metrics(_run, train_stats, epoch)

        lr_scheduler.step()
        if not np.allclose(learn_rates, lr_scheduler.get_last_lr()):
            learn_rates = lr_scheduler.get_last_lr()
            _log.info(f'Learn rates reduced to {learn_rates}.')

        # MODEL SAVING
        if args.output_dir:
            _log.info(f'Creating checkpoint for epoch [{epoch}].')
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every n epochs
            if (((epoch + 1) % args.lr_drop == 0) or
                (epoch + 1) % args.save_model_interval == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # VALIDATION
        # Validate every n epochs
        # Always evaluate first and last epoch
        if ((args.validation_freq > 0) and
            (
                (epoch == args.start_epoch) or
                ((epoch + 1) % args.validation_freq == 0) or
                (epoch == (args.epochs - 1))
            )):
            _log.info(f'Validating epoch [{epoch}].')
            val_stats = evaluate(epoch, model, criterion, data_loader_val, device, args)
            log_metrics(_run, val_stats, epoch, prefix='val_')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    _log.info(f'Finished training. Time: {total_time_str}')


def log_metrics(_run, stats: dict[str, float], epoch: int, prefix: str = ''):
    for key, value in stats.items():
        if 'frame' in key:  # Ignore frame metrics
            continue
        _run.log_scalar(prefix + key, value, epoch)



def set_seed(seed: int):
    seed += utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@ex.main
def main(_config, _run):
    args = utils.nested_dict_to_namespace(_config)
    train(args, _run)

if __name__ == '__main__':
    ex.run_commandline()
