import os
import logging
import time
from collections import OrderedDict, Counter
import copy 

import torch
from torch import autograd

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup
from detectron2.engine import default_argument_parser, hooks, HookBase
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import  get_event_storage

from detectron2.utils import comm
from detectron2.evaluation import COCOEvaluator, verify_results, inference_on_dataset, print_csv_format

from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Instances, pairwise_iou

from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.checkpoint import Checkpointer

from data.datasets import builtin

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_train_loader
import torch.utils.data as data
from detectron2.data.dataset_mapper import DatasetMapper


import torchvision.transforms as T

from mnist import  add_stn_config

logger = logging.getLogger("detectron2")



def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_optimizer(cls,cfg,model):
        params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )

        archs = ['aggregator']
        trainable = {arch: [] for arch in archs}
        
        for name,val in model.named_parameters():
            head = name.split('.')[0]
            print(head)
            if head not in archs and val.requires_grad:
                val.requires_grad = False
            elif head in archs:
                val.requires_grad = True
                trainable[head].append(val)
        
        optimizer = torch.optim.SGD(
            trainable['aggregator'],
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

        return  maybe_add_gradient_clipping(cfg, optimizer)
        

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def do_test_st():

            model = self.model 

            results = OrderedDict()
            for dataset_name in self.cfg.DATASETS.TEST:
                data_loader = build_detection_test_loader(self.cfg, dataset_name)
                evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(self.cfg.OUTPUT_DIR, "inference"))
                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)
                    storage = get_event_storage()
                    storage.put_scalar(f'{dataset_name}_AP50', results_i['bbox']['AP50'],smoothing_hint=False)
            if len(results) == 1:
                results = list(results.values())[0]
            return results
        
       
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, do_test_st))

        

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        return build_lr_scheduler(cfg, optimizer)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()

        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])



class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.
        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer) #only sgd

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)


def do_test(cfg, model, model_type=''):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
    return results

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg,model)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    for dataset_name in cfg.DATASETS.TEST:
        if 'val' in dataset_name:
            trainer.register_hooks([
                    hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD,trainer.checkpointer,f'{dataset_name}_AP50',file_prefix=f'model_best_{dataset_name}'),
                    ])
    trainer.train()




    
# def do_test(cfg, trainer,model_type=''):
#     model = trainer.model
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:
#         data_loader = build_detection_test_loader(cfg, dataset_name)
#         evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#             storage = get_event_storage()
#             storage.put_scalar(f'{dataset_name}_AP50_{model_type}', results_i['bbox']['AP50'],smoothing_hint=False)

#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results


# def main(args):
#     cfg = setup(args)
#     if args.eval_only:
#         model = Trainer.build_model(cfg)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         res = Trainer.test(cfg, model)
#         if comm.is_main_process():
#             verify_results(cfg, res)
#         return res
#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=args.resume)
#     import pdb;pdb.set_trace()
#     all_hooks  = trainer._hooks[:-1]
#     all_hooks += [hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, lambda cfg=cfg,trainer=trainer:do_test(cfg,trainer))]
#     all_hooks += [trainer._hooks[-1]]
#     trainer._hooks = all_hooks
#     trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)
    main(args)
