# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os

from yacs.config import CfgNode as CN


_C = CN()
_C.SEED = 2
_C.WORKERS = 1
_C.TRAINER = 'Trainer'


# tasks
_C.TASK = CN()
_C.TASK.NAME = 'UDA'
_C.TASK.SSDA_SHOT = 1

# ================= training ====================
_C.TRAIN = CN()
_C.TRAIN.TEST_FREQ = 150
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.SAVE_FREQ = 5000
_C.TRAIN.TTL_ITE = 8000

_C.TRAIN.BATCH_SIZE_SOURCE = 36
_C.TRAIN.BATCH_SIZE_TARGET = 36
_C.TRAIN.BATCH_SIZE_TEST = 36
_C.TRAIN.LR = 0.0002

_C.TRAIN.OUTPUT_ROOT = 'temp'
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.OUTPUT_LOG = 'log'
_C.TRAIN.OUTPUT_TB = 'tensorboard'
_C.TRAIN.OUTPUT_CKPT = 'ckpt'
_C.TRAIN.OUTPUT_RESFILE = 'log.txt'

# ================= models ====================
_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 5e-5#4
_C.OPTIM.MOMENTUM = 0.9

# ================= models ====================
_C.MODEL = CN()
_C.MODEL.PRETRAIN = False #True
_C.MODEL.BASENET = 'resent50'
_C.MODEL.BASENET_DOMAIN_EBD = False  # for domain embedding for transformer
_C.MODEL.DNET = 'Discriminator'
_C.MODEL.D_INDIM = 0
_C.MODEL.D_OUTDIM = 1
_C.MODEL.D_HIDDEN_SIZE = 1024
_C.MODEL.D_WGAN_CLIP = 0.01
_C.MODEL.VIT_DPR = 0.1
_C.MODEL.VIT_USE_CLS_TOKEN = True
_C.MODEL.VIT_PRETRAIN_EXLD = []
# extra layer
_C.MODEL.EXT_LAYER = False
_C.MODEL.EXT_NUM_TOKENS = 100
_C.MODEL.EXT_NUM_LAYERS = 1
_C.MODEL.EXT_NUM_HEADS = 24
_C.MODEL.EXT_LR = 10.
_C.MODEL.EXT_DPR = 0.1
_C.MODEL.EXT_SKIP = True
_C.MODEL.EXT_FEATURE = 768

# ================= dataset ====================
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.NUM_CLASSES = 2 #10
_C.DATASET.NAME = 'binary' #['binary' '6class'] #'office_home'
_C.DATASET.SOURCE = []
_C.DATASET.TARGET = []
_C.DATASET.TRIM = 0

# ================= method ====================
_C.METHOD = CN()
_C.METHOD.W_ALG = 1.0
_C.METHOD.ENT = False

# HDA
_C.METHOD.HDA = CN()
_C.METHOD.HDA.W_HDA = 1.0
_C.METHOD.HDA.LR_MULT = 1.0  # set as 5.0 to tune the lr_schedule to follow the setting of original HDA


def get_default_and_update_cfg(args):
    cfg = _C.clone()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    #
    cfg.SEED = args.seed

    #
    if args.data_root:
        cfg.DATASET.ROOT = args.data_root

    # dataset maps
    maps = {
        'office_home': {
            'p': 'product',
            'a': 'art',
            'c': 'clipart',
            'r': 'real_world'
        },
        '6class': {
            'b': 'Biop',
            'f': 'Foveal',
            'n': 'bbank',
        },
        'binary': {
            'b': 'Biop',
            'bi':'biop',
            'f': 'Foveal',
            'n': 'bbank',
            'j':'bbankjpg',
            'rb': 'revised_biobankjpg',
            'ukb':'biobankLEjpg'
        },
    }

    # MSDA
    if cfg.TASK.NAME == 'MSDA':
        args.source = [k for k in maps[cfg.DATASET.NAME].keys()]
        args.source.remove(args.target[0])

    # source & target
    cfg.DATASET.SOURCE = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in args.source]
    cfg.DATASET.TARGET = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in args.target]

    # class
    if cfg.DATASET.NAME == 'office_home':
        cfg.DATASET.NUM_CLASSES = 65
    elif cfg.DATASET.NAME == '6class':
        cfg.DATASET.NUM_CLASSES = 6
    elif cfg.DATASET.NAME == 'binary':
        cfg.DATASET.NUM_CLASSES = 2
    else:
        raise NotImplementedError(f'cfg.DATASET.NAME: {cfg.DATASET.NAME} not imeplemented')

    # output
    if args.output_root:
        cfg.TRAIN.OUTPUT_ROOT = args.output_root
    if args.output_dir:
        cfg.TRAIN.OUTPUT_DIR = args.output_dir
    else:
        cfg.TRAIN.OUTPUT_DIR = '_'.join(cfg.DATASET.SOURCE) + '2' + '_'.join(cfg.DATASET.TARGET) + '_' + str(args.seed)

    #
    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_TB = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'tensorboard', cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_TB, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    return cfg


def check_cfg(cfg):
    # OUTPUT
    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_TB = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'tensorboard', cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_TB, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    # dataset
    maps = {
        'office_home': {
            'p': 'product',
            'a': 'art',
            'c': 'clipart',
            'r': 'real_world'
        },
        '6class': {
            'b': 'Biop',
            'f': 'Foveal',
            'n': 'bbank'
        },
        'binary': {
            'b': 'Biop',
            'bi':'biop',
            'f': 'Foveal',
            'n':'bbank',
            'j':'bbankjpg',
            'rb': 'revised_biobankjpg',
            'ukb': 'biobankLEjpg'
        }
        
    }
    cfg.DATASET.SOURCE = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in cfg.DATASET.SOURCE]
    cfg.DATASET.TARGET = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in cfg.DATASET.TARGET]

    datapath_list = {
        'office-home': {
            'p': ['Product.txt', 'Product.txt'],
            'a': ['Art.txt', 'Art.txt'],
            'c': ['Clipart.txt', 'Clipart.txt'],
            'r': ['Real_World.txt', 'Real_World.txt']
        },
        '6class': {
            'b': ['Biop_train.txt','Biop_valid.txt'],
            'f': ['Foveal_train.txt','Foveal_valid.txt']
        },
        'binary': {
            'b': ['Biop_train_bin.txt','Biop_valid_bin.txt'],
            'bi': ['biop_train_bin.txt','biop_valid_bin.txt'],
            'f': ['Foveal_train_bin.txt','Foveal_valid_bin.txt'],
            'n': ['bbank_train_bin.txt','bbank_train_bin.txt'],
            'j': ['bbankjpg_train_bin.txt','bbankjpg_train_bin.txt'],
            'rb': ['revised_biobankjpg_train_bin.txt','revised_biobankjpg_valid_bin.txt'], # 
            'ukb' : ['revised_biobankjpg_train_bin.txt','biobankLEjpg.txt']
        }
    }

    # class
    if cfg.DATASET.NAME == 'office_home':
        cfg.DATASET.NUM_CLASSES = 65
    elif cfg.DATASET.NAME == '6class':
        cfg.DATASET.NUM_CLASSES = 6
    elif cfg.DATASET.NAME == 'binary':
        cfg.DATASET.NUM_CLASSES = 2
    else:
        raise NotImplementedError(f'cfg.DATASET.NAME: {cfg.DATASET.NAME} not imeplemented')

    return cfg
