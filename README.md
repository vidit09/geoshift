# Learning Transformation to reduce Geometric Shift in Object Detection
[ [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Vidit_Learning_Transformations_To_Reduce_the_Geometric_Shift_in_Object_Detection_CVPR_2023_paper.pdf) ]




### Installation
This code is based on [detectron2](https://github.com/facebookresearch/detectron2) and requires python>= 3.6
```bash
pip install -r requirements.txt
```

###  Dataset 
Set the environ variable DETECTRON2_DATASETS to the parent folder of the datasets
```
/datasets
    /cityscapes
    /kitti
    /mot
```    
Download    
1. Cityscapes & Kitti  from -- https://github.com/chengchunhsu/EveryPixelMatters#dataset 

2. MOT sequence MOT20-02 -- https://motchallenge.net

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

### Citation
```bibtex
@inproceedings{vidit2023learning,
  title={Learning Transformations To Reduce the Geometric Shift in Object Detection},
  author={Vidit, Vidit and Engilberge, Martin and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17441--17450},
  year={2023}
}
```
