# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import logging
import torch


_log = logging.getLogger()


def load_model(model: torch.nn.Module,
               model_path: str,
               resume: bool = False,
               optimizer: torch.optim.Optimizer = None,
               lr_scheduler: torch.optim.lr_scheduler.StepLR = None,
               lr_drop: int = 0):

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if resume:
        _log.info(f'Resuming model \"{model_path}\"')
    else:
        _log.info(f'Loaded model \"{model_path}\"')

    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    warnings = 0
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                _log.warn(f'Skip loading parameter {k}, required shape {model_state_dict[k].shape}, '
                          f'loaded shape {state_dict[k].shape}.')
                warnings += 1
                if 'class_embed' in k:
                    _log.warn(f'Load class_embed: {k} shape={state_dict[k].shape}')
                    if model_state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        state_dict[k] = state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        state_dict[k] = state_dict[k][1:4]
                    else:
                        raise NotImplementedError(f'Invalid shape: {model_state_dict[k].shape}')
                    continue
                state_dict[k] = model_state_dict[k]
        else:
            _log.warn(f'Drop parameter {k}.')
            warnings += 1
    for k in model_state_dict:
        if k not in state_dict:
            _log.warn(f'No parameter {k}.')
            warnings += 1
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    if warnings > 0:
        _log.warn('If you see this, your model does not fully load the pre-trained weight. '
                  'Please make sure you set the correct number of classes for your dataset.')

    if not resume:
        return model

    start_epoch = 0

    if (optimizer is not None and 'optimizer' in checkpoint and
        lr_scheduler is not None and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint):
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        _log.info('Loaded optimizer.')

        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if lr_scheduler.step_size != lr_drop:
            lr_scheduler.step_size = lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            _log.info(f'Learn rate drop every {lr_scheduler.step_size} has been overwritten to {lr_drop}.')

        start_epoch = checkpoint['epoch'] + 1

        _log.info(f'Resumed optimizer with start lr {lr_scheduler.get_last_lr()[0]:.0e}.')
    else:
        _log.info('No optimizer available to load.')

    return model, optimizer, lr_scheduler, start_epoch
