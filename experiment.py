import logging
import os

import sacred
from sacred import SETTINGS
from sacred.observers import FileStorageObserver


def new_experiment(name: str, training: bool, *,
                   save_gpu_info: bool = True,
                   save_cpu_info: bool = True,
                   save_git_info: bool = True) -> sacred.Experiment:

    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0

    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - [%(levelname)s] (Rank: {rank}) - %(message)s')

    SETTINGS['HOST_INFO']['CAPTURED_ENV'].append('CUDA_LAUNCH_BLOCKING')
    SETTINGS['HOST_INFO']['CAPTURED_ENV'].append('RANK')
    SETTINGS['HOST_INFO']['CAPTURED_ENV'].append('WORLD_SIZE')
    SETTINGS['HOST_INFO']['CAPTURED_ENV'].append('LOCAL_RANK')

    SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = save_gpu_info
    SETTINGS['HOST_INFO']['INCLUDE_CPU_INFO'] = save_cpu_info

    ex = sacred.Experiment(name, save_git_info=save_git_info)
    if training:
        ex.observers.append(FileStorageObserver('runs', copy_sources=False))

    ex.add_config(os.path.join('configs', 'base', 'misc.yaml'))
    ex.add_config(os.path.join('configs', 'base', 'model.yaml'))
    ex.add_config(os.path.join('configs', 'base', 'dataset.yaml'))

    return ex
