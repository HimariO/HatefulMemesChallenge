import _init_paths

import os
import argparse
import subprocess

import torch
import fire
from loguru import logger

from cls.function.config import config, update_config
from cls.function.train import train_net
from cls.function.test import test_net
from cls.data import build


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set',
                        default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')

    # easy test pretrain model
    parser.add_argument('--partial-pretrain', type=str)
    parser.add_argument('--k-fold-i', type=int)

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    if args.partial_pretrain is not None:
        config.NETWORK.PARTIAL_PRETRAIN = args.partial_pretrain

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config


def main():
    args, config = parse_args()

    if config.DATASET.K_FOLD:
        outputpath = config.OUTPUT_PATH
        
        i = args.k_fold_i
        config.DATASET.K_FOLD_N = i
        
        out_base = os.path.basename(outputpath)
        new_output_path = os.path.join(
            os.path.dirname(outputpath),
            f"{out_base}-fold-{i}"
        )
        config.OUTPUT_PATH = new_output_path
        
        rank, model = train_net(args, config)
        if args.do_test and (rank is None or rank == 0):
            test_net(args, config)
    else:
        rank, model = train_net(args, config)
        if args.do_test and (rank is None or rank == 0):
            test_net(args, config)

def test():
    args, config = parse_args()
    train_loader = build.make_dataloader(config, mode='train', distributed=False)
    test_net(args, config)
    
    for data in train_loader:
        print(data)
        break
    
    # import pdb; pdb.set_trace()
    # rank, model = train_net(args, config)


if __name__ == '__main__':
    with logger.catch(reraise=True):
        main()
        # test()


