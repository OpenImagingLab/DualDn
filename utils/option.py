import yaml
import torch
import random
import argparse
import numpy as np

from os import path as osp
from collections import OrderedDict
from .dist import get_dist_info, init_dist, master_only

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

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

def yaml_load(opt_file, test_mode):
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
    opt['root_path'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    if opt['is_train']:  # train mode
        experiments_root = osp.join(opt['root_path'], 'experiments',opt['name'])
        opt['path']['log'] = experiments_root
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
    else:  # test mode
        results_root = osp.join(opt['root_path'], 'results', opt['name'])
        opt['path']['log'] = results_root
        opt['path']['results_root'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, default=r'./options/DualDn.yml', help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true', default=False, help='Auto-resume switch')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--test', action='store_true', default=False, help='Test/Train mode')
    parser.add_argument('--local_rank', type=int, default=0) 
    parser.add_argument('--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    ##* load and update yaml option file
    opt = yaml_load(args.opt, args.test)
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

    ##* auto resume 
    if opt['path']['resume_state'] == 'auto_resume':
        opt['auto_resume'] = True
    else:
        opt['auto_resume'] = args.auto_resume

    ##* random seed
    if opt['manual_seed'] is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    seed = opt.get('manual_seed')
    for _, dataset in opt['datasets'].items():
        dataset['seed'] = seed
    set_random_seed(seed + opt['rank'])

    ##* debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']
    if 'debug' in opt['name']:  # change some options for debug mode
        if 'val' in opt:
            opt['val']['val_freq'] = 8
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 8

    ##* force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
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

@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


