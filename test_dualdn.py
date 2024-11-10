# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import os
import yaml
import torch
import random
import logging
import argparse
import numpy as np
from copy import deepcopy
from os import path as osp
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from models import build_model
from data import build_dataset, build_dataloader
from utils import get_root_logger, get_time_str, make_exp_dirs, dict2str, get_dist_info, init_dist

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(opt_file, val_datasets, test_mode, iters):
    """Load yaml file or string.

    Args:
        f (str): Option file path or a python string.

    Returns:
        (dict): Options.
    """
    if osp.isfile(opt_file):
        with open(opt_file, mode='r') as f:
            loaded_opt = yaml.load(f, Loader=ordered_yaml()[0])
    else:
        loaded_opt = yaml.load(opt_file, Loader=ordered_yaml()[0])
    loaded_opt['is_train'] = not test_mode
    loaded_opt['opt_path'] = opt_file
    loaded_opt['iters'] = iters

    loaded_opt['datasets']['val']['val_datasets']['Synthetic']['mode'] = True if val_datasets == 'Synthetic' else False
    loaded_opt['datasets']['val']['val_datasets']['Real_captured']['mode'] = True if val_datasets == 'Real_captured' else False
    loaded_opt['datasets']['val']['val_datasets']['DND']['mode'] = True if val_datasets == 'DND' else False

    return loaded_opt

def opt_update(opt):
    """Update option file or string for individent user.
    Args:
        opt(dict): Loaded Options dict.

    Returns:
        (dict): Updated Options dict.
    """

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        if dataset.get('data_path') is not None:
            dataset['data_path'] = osp.expanduser(dataset['data_path'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    # add root path
    opt['root_path'] = osp.abspath(osp.join(__file__, osp.pardir))
    opt['experiment_path'] = osp.abspath(osp.join(opt['opt_path'], osp.pardir))

    results_root = osp.join(opt['root_path'], 'results', opt['name'])
    opt['path']['log'] = results_root
    opt['path']['results_root'] = results_root
    opt['path']['visualization'] = results_root
    opt['path']['pretrain_network'] = osp.join(opt['experiment_path'], 'models', 'net_g_{}.pth'.format(opt['iters']))
    opt['path']['param_key'] = 'param_key'

    return opt

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, default=r'./experiments/DualDn_Big/denoising.yml', help='Path to option YAML file.')
    parser.add_argument('--num_iters', type=int, default=300000, help='num iterations')
    parser.add_argument('--val_datasets', choices=['Synthetic', 'Real_captured', 'DND'], help='val_datasets for testing')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--test', default=True, help='Test/Train mode')
    parser.add_argument('--local_rank', type=int, default=0) 
    parser.add_argument('--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    ##* load and update yaml option file
    opt = yaml_load(args.opt, args.val_datasets, args.test, args.num_iters)
    opt = opt_update(opt)

    ##* distributed-training settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
        print('init dist .. ', args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    ##* random seed
    if opt['manual_seed'] is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    seed = opt.get('manual_seed')
    for _, dataset in opt['datasets'].items():
        dataset['seed'] = seed
    set_random_seed(seed + opt['rank'])

    ##* force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    return opt

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options()

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'val':
            for dataset_type, dataset in dataset_opt['val_datasets'].items():
                if dataset['mode']:
                    dataset_opt['dataset_type'] = dataset_type
                    dataset_opt['data_path'] = dataset['data_path']

                    if dataset_type == 'Synthetic':
                        noise_levels, alphas = [], []
                        if type(dataset_opt['syn_noise']['noise_level']) == list:
                            noise_levels = deepcopy(dataset_opt['syn_noise']['noise_level'])
                        else:
                            noise_levels[0] = deepcopy(dataset_opt['syn_noise']['noise_level'])
                        if type(dataset_opt['syn_isp']['alpha']) == list:
                            alphas = deepcopy(dataset_opt['syn_isp']['alpha'])
                        else:
                            alphas[0] = deepcopy(dataset_opt['syn_isp']['alpha'])
                        for _, noise_level in enumerate(noise_levels):
                            for _, alpha in enumerate(alphas):
                                dataset_opt['syn_noise']['noise_level'] = deepcopy(noise_level)
                                dataset_opt['syn_isp']['alpha'] = deepcopy(alpha)
                                val_set = build_dataset(dataset_opt)
                                test_loader = build_dataloader(
                                    val_set,
                                    dataset_opt,
                                    num_gpu=opt['num_gpu'],
                                    dist=opt['dist'],
                                    sampler=None,
                                    seed=opt['manual_seed'])
                                logger.info(
                                    f'Number of val images in {dataset_type} {dataset_opt["name"]}: '
                                    f'{len(val_set)}')
                                test_loaders.append(test_loader)
                    else:
                        val_set = build_dataset(dataset_opt)
                        test_loader = build_dataloader(
                            val_set,
                            dataset_opt,
                            num_gpu=opt['num_gpu'],
                            dist=opt['dist'],
                            sampler=None,
                            seed=opt['manual_seed'])
                        logger.info(
                                    f'Number of val images in {dataset_type} {dataset_opt["name"]}: '
                                    f'{len(val_set)}')
                        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.dataset_type
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, -1, None)


if __name__ == '__main__':
    main()
