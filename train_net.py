import os

from detectron2 import model_zoo

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch, default_setup
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer


from data.datasets import builtin
from mnist import  add_stn_config


from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results

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

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

