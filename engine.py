# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import logging
import math
import sys
import time
from argparse import Namespace
from typing import Iterable

import torch

import util.misc as utils
from datasets.data_prefetcher import data_dict_to_device
from models.clip_matcher import ClipMatcher

_log = logging.getLogger()

def train_one_epoch(epoch: int, model: torch.nn.Module, criterion: ClipMatcher,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, args: Namespace):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = f'Epoch [{epoch}]'

    for data_dict in metric_logger.log_every(data_loader, args.stats_print_freq, header, show_gpu_info=args.print_gpu_info):
        data_dict = data_dict_to_device(data_dict, device, non_blocking=True)

        sample_size = len(data_dict['gt_instances'])
        if sample_size != args.sample_size:
            _log.warning(f'Current iteration has a different sample size ({sample_size}) than configured ({args.sample_size}).')


        # Prediction (also already calculates loss)
        start_model_time = time.time()
        outputs, metrics = model(data_dict)
        metrics['time_model'] = time.time() - start_model_time

        start_loss = time.time()
        loss_dict = criterion(outputs)
        losses = sum(criterion.scale_loss(loss_dict).values())
        metrics['time_loss'] = time.time() - start_loss

        # reduce losses over all GPUs for logging purposes
        state = utils.reduce_dict_async(loss_dict)

        start_backward = time.time()
        optimizer.zero_grad()
        losses.backward()
        metrics['time_backward'] = time.time() - start_backward

        if args.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

        start_optimizer = time.time()
        optimizer.step()
        metrics['time_optimizer'] = time.time() - start_optimizer

        start_loss = time.time()
        loss_dict_reduced = utils.reduce_dict_await(state)
        loss_dict_reduced_scaled = criterion.scale_loss(loss_dict_reduced)
        loss_reduced = sum(loss_dict_reduced_scaled.values()).item()
        metrics['time_loss'] += (time.time() - start_loss)

        if not math.isfinite(loss_reduced):
            _log.critical(f'Loss is {loss_reduced}, stopping training. {loss_dict_reduced}')
            sys.exit(1)

        metric_logger.update(loss=loss_reduced, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(**metrics)

    metric_logger.synchronize_between_processes()
    _log.info(f'Averaged stats: {metric_logger}')

    return metric_logger.get_global_avg_metrics()

@torch.no_grad()
def evaluate(epoch: int, model: torch.nn.Module, criterion: ClipMatcher,
             data_loader: Iterable, device: torch.device, args: Namespace):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('gpu_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = f'Val Epoch [{epoch}]'

    losses: list[dict[str, torch.Tensor]] = []

    # TODO: Optimize data loading

    for data_dict in metric_logger.log_every(data_loader, args.val_stats_print_freq, header, loss_meter='gpu_loss', show_gpu_info=args.print_gpu_info):
        data_dict = data_dict_to_device(data_dict, device, non_blocking=True)

        # Prediction (also already calculates loss)
        start_model_time = time.time()
        outputs, metrics = model(data_dict)
        metrics['time_model'] = time.time() - start_model_time

        start_loss = time.time()
        loss_dict = criterion(outputs)
        loss = sum(criterion.scale_loss(loss_dict).values()).item()
        metrics['time_loss'] = time.time() - start_loss

        losses.append(loss_dict)

        metric_logger.update(gpu_loss=loss)
        metric_logger.update(**metrics)

    _log.info(f'Reducing loss.')
    # Remove previous loss meter
    # We will add a new meter which logs the reduced loss
    metric_logger.del_meter('gpu_loss')

    # reduce losses over all GPUs for logging purposes
    for loss_dict in losses:
        agg_loss_dict = criterion.sum_frame_loss(loss_dict)
        agg_loss_dict_reduced = utils.reduce_dict(agg_loss_dict)
        loss_reduced = sum(criterion.scale_loss(agg_loss_dict_reduced).values()).item()

        metric_logger.update(loss=loss_reduced, **agg_loss_dict_reduced)

    metric_logger.synchronize_between_processes()
    _log.info(f'Averaged validation stats: {metric_logger}')

    return metric_logger.get_global_avg_metrics()
