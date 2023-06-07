from detectron2.config import  CfgNode as CN

def add_stn_config(cfg):

    cfg.MODEL.TPS_ARCH = ""
    cfg.MODEL.STN_ARCH = ""
    cfg.MODEL.DIS_ARCH = ""
    cfg.MODEL.ALTFREQ  = 2000 # ~1epoch for 2batchsize
    cfg.TEST.EVAL_SAVE_PERIOD  = 2000
    cfg.MODEL.WEIGHTS_BACKBONE = ''
    cfg.MODEL.HOMOGRAPHY_PREDICTOR_WEIGHT = ""
    cfg.SOLVER.STN_LR = 0.01
    cfg.SOLVER.DIS_ENABLE_ITER = 1000
    cfg.MODEL.META_ARCH_AGG = ""
    