import numpy as np
import torch

import os
import argparse
import sys

import utils

import wandb

 
from configs.config import get_cfg_defaults

import train
import train_frame

from model.video_models import video_models, i3d, tsm, tsm_seg, tsm_unet_seg
from model import unet

### CUDA Debug Flags ###
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_args():
    """
    [Code refered from FAIR's SlowFast codebase - https://github.com/facebookresearch/SlowFast/blob/9839d1318c0ae17bd82c6a121e5640aebc67f126/slowfast/utils/parser.py#L13]
    Parse the following arguments for a default parser for LungUS AI users.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide LungUS AI video training and testing pipeline."
    )
    
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="code/configs/exp.yaml",
        type=str,
    )
    
    parser.add_argument(
        "opts",
        help="See code/config/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def loadConfig():
    
    args = parse_args()

    # Load default config
    cfg = get_cfg_defaults()
    

    # Merge Experiment config
    # cfg.merge_from_file("experiment.yaml")
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    
    

    # Merge cmd args
    # opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
    # cfg.merge_from_list(opts)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)



    # Add constuctive configs
    cfg.EXPERIMENT.ROOT_DIR = os.path.join("results", cfg.EXPERIMENT.NAME)
    cfg.EXPERIMENT.RUN_NAME = f"{cfg.EXPERIMENT.MODEL}_{cfg.EXPERIMENT.DATASET}_{cfg.EXPERIMENT.TRIAL}"
    cfg.EXPERIMENT.FOLD_NAME = f"{cfg.EXPERIMENT.RUN_NAME}_{cfg.DATA.VAL_FOLDS[0]}"
    cfg.EXPERIMENT.DIR = os.path.join(cfg.EXPERIMENT.ROOT_DIR, cfg.EXPERIMENT.FOLD_NAME)
    # cfg.EXPERIMENT.ANALYSIS_DIR = os.path.join(cfg.EXPERIMENT.DIR, "videoGradCAM")
    cfg.EXPERIMENT.GRAD_CAM_DIR = os.path.join(cfg.EXPERIMENT.DIR, "videoGradCAM")
    cfg.EXPERIMENT.CHECKPOINT_DIR = os.path.join(cfg.EXPERIMENT.DIR, "checkpoints")

    # Freeze the config
    cfg.freeze()

    print(cfg)
    
    return cfg


def createExpDirs(cfg):

    if cfg.EXPERIMENT.MODE == "Train":

        if cfg.EXPERIMENT.DEBUG:
            utils.removeDir(cfg.EXPERIMENT.DIR)

        utils.createDir(cfg.EXPERIMENT.DIR)
    
    # #Create tensorboard dir
    # utils.createDir(constants.tensorboard_dir_path, exist_ok=True)


def initModel(cfg):
    if cfg.EXPERIMENT.MODEL == "tsm": #### chosen one for the currect exp I am doing (89% one)
        model = tsm.TSN(
                    num_class = cfg.DATA.NUM_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS
                )       
    else:
        raise ValueError(f"Unsupported cfg.EXPERIMENT.MODEL = {cfg.EXPERIMENT.MODEL}!")

    return model


def main():
    
    #Load Config
    cfg = loadConfig()
    #Create exp dir
    createExpDirs(cfg)

    #Write the config
    with open(os.path.join(cfg.EXPERIMENT.DIR, "config.yaml"), "w") as file1:
        file1.write(str(cfg))
 
    model = initModel(cfg)

    if cfg.MODEL.TYPE == "video":    # into this
        train.cli_main(cfg, model)
    elif cfg.MODEL.TYPE == "frame":    
        train_frame.cli_main(cfg, model)
    else:
        raise ValueError(f"Unsupported cfg.MODEL.TYPE = {cfg.MODEL.TYPE}!")
    
if __name__ == "__main__":

    print("Started...")
    main()
    print("Finished!")