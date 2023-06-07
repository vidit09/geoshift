# Learning transformation to reduce geometric shift
This code is based on detectron2 -- https://github.com/facebookresearch/detectron2  

### Installation
Requires python>= 3.6
```bash
pip install -r requirements.txt
```

###  Dataset 
set the environ variable DETECTRON2_DATASETS to the parent folder of the datasets
/datasets
    /cityscapes
    /kitti
    /mot
cityscapes & kitti were used from -- https://github.com/chengchunhsu/EveryPixelMatters#dataset 
mot -- https://motchallenge.net , sequence MOT20-02

### Base Training
```bash
python train_net.py --config-file configs/<file_name>_random_crop.yaml 
.py
```
### Aggregator Training
```bash
python train_net_only_aggregator.py --config-file configs/<file_name>_aggregator_fivestnperspective.yaml 
.py
```

### Mean Teacher Training
We train on single V100 GPU with batch size 2 for mean teacher(in config setting which means 2 source domain and 2 target domain), steps 
```bash
python train_net_student_teacher_<task>.py --config-file configs/<task>_student_teacher.yaml SOLVER.BASE_LR 1e-3 SOLVER.STEPS [10000,] MODEL.STN_ARCH FIVE_OPT_PERSPECTIVE
.py
```