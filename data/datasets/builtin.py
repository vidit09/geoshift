# Copyright (c) Facebook, Inc. and its affiliates.
from .cityscapes_car import register_all_cityscapes
from .cityscapes_person import register_all_cityscapes as register_city_person
from .kitti_car  import register_dataset as register_kitti_dataset
from .mot_person import register_dataset as register_mot_person

import os 

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
DEFAULT_DATASETS_ROOT = "data/"

register_city_person(_root)
register_all_cityscapes(_root)
register_kitti_dataset(_root)
register_mot_person(_root)
