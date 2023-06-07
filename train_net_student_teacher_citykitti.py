from cgi import parse_multipart
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

from mnist import add_stn_config

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

class CombineLoaders(data.IterableDataset):
    def __init__(self,loaders):
        self.loaders = loaders

    def __iter__(self,):
        dd = iter(self.loaders[1])
        for d1 in self.loaders[0]:
            try:
                d2 = next(dd)
            except:
                dd=iter(self.loaders[1])
                d2 = next(dd)

            list_out_dict=[]
            for v1,v2 in zip(d1,d2):
                out_dict = {}
                for k in v1.keys():
                    out_dict[k] = (v1[k],v2[k])
                list_out_dict.append(out_dict)

            yield list_out_dict

class StudentTeacher(torch.nn.Module):
    def __init__(self,student_model,teacher_model) -> None:
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "student": student_model,
            "teacher": teacher_model,
        })

    def get(self,model_kind):
        return self.model[model_kind]

    def forward(self,model_kind,data):
        self.model[model_kind](data)

class Trainer(DefaultTrainer):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.teach_model = None
        self.aug = T.ColorJitter(brightness=.5, hue=.3)
        self.alternatefreq = cfg.MODEL.ALTFREQ
    
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        # model = StudentTeacher(student_model=model_student,teacher_model=model_teacher)
        return model

    @classmethod
    def build_train_loader(cls,cfg):
        original  = cfg.DATASETS.TRAIN
        print(original)
        cfg.DATASETS.TRAIN=(original[0],)
        data_loader1 = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        cfg.DATASETS.TRAIN=(original[1],)
        data_loader2 = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        cdata = CombineLoaders([data_loader1,data_loader2])
        return cdata

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_optimizer(cls,cfg,model):
        
        archs = ['stn','others']
        trainable = {arch: [] for arch in archs}
        
        for name,val in model.named_parameters():
            head = name.split('.')[0]
            # print(head)
            if head in ['stn']:
                val.requires_grad = True   
                trainable[head].append(val)   
            else:
                val.requires_grad = True
                trainable['others'].append(val)   

        paramlist = []
        for v in trainable.values():
            paramlist += v
        assert len(paramlist) == sum([len(v) for v in trainable.values()])

        optimizer1 = torch.optim.Adam(
            trainable['stn'],
            cfg.SOLVER.BASE_LR,
        )

        optimizer2 = torch.optim.SGD(
            trainable['others'],
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return (maybe_add_gradient_clipping(cfg, optimizer1),maybe_add_gradient_clipping(cfg, optimizer2))

    def EMA(self,model,teach_model,alpha):
        x = zip(model.named_parameters(),teach_model.named_parameters())
        for d_stud,d_teach in x:
            name,value = d_stud
            name,teach_val = d_teach

            if value.requires_grad:
                teach_val.data.mul_(alpha)
                teach_val.data.add_(value.detach()*(1-alpha))

        x = zip(model.named_buffers(),teach_model.named_buffers())
        for d_stud,d_teach in x:
            names,value = d_stud
            namet,teach_val = d_teach

            # print(names,namet)
            #TODO check if these layers have requires grad
            try:
                if 'num_batches_tracked' not in names:
                    teach_val.data.mul_(alpha)
                    teach_val.data.add_(value.detach()*(1-alpha))
            except:
                import pdb;pdb.set_trace()
                
                

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start


        """
        If you want to do something with the losses, you can wrap the model.
        """
        data_s=[]
        data_t=[]
        for ins in data:
            dict_s={}
            dict_t={}
            for k in ins.keys():
                dict_s[k],dict_t[k]=ins[k]
            data_s.append(dict_s)   
            data_t.append(dict_t)
        
        # train student with source labels
        for ind, d in enumerate(data_s):
            d['image'] = self.aug(d['image'].cuda())
        loss_dict_s = self.model(data_s)

        # generate pseudo labels
        with torch.no_grad():
            self.teach_model.eval()
            results = self.teach_model.inference(data_t,do_postprocess=False)   
            # results = [r.to('cpu') for r in results]
            teach_data = []
            for ind, r in enumerate(results):
                pdata= {}
                keep = r.scores>0.6
                pruned = r[keep]
                for k,v in data_t[ind].items():
                    ## there is mismatch in the data_t image size and result instance size
                    if k == 'instances':
                        inst = Instances(image_size=pruned.image_size,\
                            gt_boxes=pruned.pred_boxes.to('cpu'),\
                                gt_classes=pruned.pred_classes.cpu())
                        
                        pdata.update({k:inst})
                    elif k == 'image':
                        
                        pdata.update({k:self.aug(v.cuda())})
                    else:
                        pdata.update({k:v})
                teach_data.append(pdata)
            # del results
            # import gc;gc.collect();torch.cuda.empty_cache()

        #pass the labels to the student
        loss_dict_t = self.model(teach_data)
        
        loss_dict = {}

        l1 = 0 
        for k,v in loss_dict_t.items():
            #loss_dict.update({k+'_t':v})
            if k in ['loss_rpn_loc','loss_box_reg']:
                continue
            l1 += v
        l1 = 0.01*l1
    
        l2 = 0 
        for k,v in loss_dict_s.items():
            #loss_dict.update({k+'_s':v})
            l2 += v

        l1 = l1+l2        

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer[0].zero_grad() #stn
        self.optimizer[1].zero_grad() #others

        l1.backward()   

        self.optimizer[0].step() #stn
        if self.iter > 10000:
            self.optimizer[1].step() #others
            # print('other')
        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        self.EMA(self.model,self.teach_model,alpha=0.99)
        
        for k,v in loss_dict_s.items():
            loss_dict.update({k+'_s':v})
            loss_dict.update({k+'_t':loss_dict_t[k]})
        
        self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """

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

        def do_test_st(flag):
            if flag == 'st':
                model = self.model 
            elif flag == 'te':
                model = self.teach_model
            else:
                print("Error in the flag")

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
                    storage.put_scalar(f'{dataset_name}_AP50_{flag}', results_i['bbox']['AP50'],smoothing_hint=False)
            if len(results) == 1:
                results = list(results.values())[0]
            return results
        
       
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, lambda flag='st': do_test_st(flag)))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, lambda flag='te': do_test_st(flag)))

        

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

        return build_lr_scheduler(cfg, optimizer[1])

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer1"] = self.optimizer[0].state_dict()
        ret["optimizer2"] = self.optimizer[1].state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer[0].load_state_dict(state_dict["optimizer1"])
        self.optimizer[1].load_state_dict(state_dict["optimizer2"])



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
        self._best_param_group_id1 = LRScheduler.get_best_param_group_id(self._optimizer[1]) #only sgd
        self._best_param_group_id2 = LRScheduler.get_best_param_group_id(self._optimizer[0])

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
        lr1 = self._optimizer[1].param_groups[self._best_param_group_id1]["lr"]
        self.trainer.storage.put_scalar("lr1", lr1, smoothing_hint=False)
        lr2 = self._optimizer[0].param_groups[self._best_param_group_id2]["lr"]
        self.trainer.storage.put_scalar("lr2", lr2, smoothing_hint=False)
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
    trainer.teach_model = copy.deepcopy(trainer.model)
    for dataset_name in cfg.DATASETS.TEST:
        if 'val' in dataset_name:
            trainer.register_hooks([

                    hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD,trainer.checkpointer,f'{dataset_name}_AP50_st',file_prefix='model_best_st'),
                    hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD,DetectionCheckpointer(trainer.teach_model, save_dir=cfg.OUTPUT_DIR)
                                                ,f'{dataset_name}_AP50_te',file_prefix='model_best_te')
                    ])

    trainer.train()
    DetectionCheckpointer(trainer.teach_model,save_dir=cfg.OUTPUT_DIR).save('model_teach_final')
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)

    main(args)
