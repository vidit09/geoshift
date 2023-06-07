import os
import errno

from tqdm import tqdm
import pickle as pkl
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

mot_class_name = ['person']

def get_annotation(root, bboxs, ind, imagefolder, frameid):
    
    impath = os.path.join(imagefolder, frameid.zfill(6)+'.jpg')
    im = cv2.imread(impath)
    assert im is not None
    
    record = {}
    record["file_name"] = impath
    record["image_id"] = ind
    record["annotations"] = []

    for box in bboxs:

        record_obj = {
        "bbox": box,
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": 0,
        }
        record["annotations"].append(record_obj)


    if len(record["annotations"]):
        record["height"] = im.shape[0]
        record["width"] = im.shape[1]
        return record
    else:
        return None

def MOTforDetectron(root,split,setfile,imagefolder):

    cache_dir = os.path.join(root, 'cache')

    pkl_path = os.path.join(cache_dir,split+'.pkl')

    if os.path.exists(pkl_path):
        with open(pkl_path,'rb') as f:
            return pkl.load(f)
    
    dataset_dicts = []

    all_frames={}
    # import pdb;pdb.set_trace()
    with open(setfile,'r') as f:
        for lines in f:
            frameid,personid,l,t,w,h,cls,_,_ = lines.split(',')
            if cls == '1':
                t,l,w,h = int(t),int(l),int(w),int(h)
                if frameid in all_frames.keys():
                    all_frames[frameid].append([l-1,t-1,w+l-1,h+t-1])
                else:
                    all_frames.update({frameid:[[l-1,t-1,w+l-1,h+t-1]]})

    annind = 0
    for frameid, bboxs in all_frames.items():

        record = None
        if split == 'train' and int(frameid) <= 2000:
            record = get_annotation(root,bboxs,annind,imagefolder,frameid)
            annind += 1
        elif split == 'val' and int(frameid) > 2000:
            record = get_annotation(root,bboxs,annind,imagefolder,frameid)
            annind += 1
        
        if record is not None:
            dataset_dicts.append(record)

    with open(pkl_path, 'wb') as f:
        pkl.dump(dataset_dicts,f)
    return dataset_dicts


def register_dataset(datasets_root):
    dataset_list = ['MOT20']
    
    for d in dataset_list:
        try:
            cache_dir = os.path.join(datasets_root,d,'cache')
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            pass

    settype = ['train','val'] #{'train':None,'val':None}
    
    image_sets_files = []
    image_folders = []

    for d in dataset_list:
        image_sets_files.append(os.path.join(datasets_root,d, "train", "MOT20-02", "gt",'gt.txt'))
        image_folders.append(os.path.join(datasets_root,d, "train", "MOT20-02",'img1'))
       
    for ind,name in enumerate(dataset_list):
        for d in settype:

            imsetfile  = image_sets_files[ind]
            imagefolder = image_folders[ind]
            DatasetCatalog.register(name+"_" + d, lambda datasets_root=datasets_root,name=name,d=d, \
                imsetfile=imsetfile, imagefolder=imagefolder: MOTforDetectron(os.path.join(datasets_root,name), d, imsetfile, imagefolder))

            MetadataCatalog.get(name+ "_" + d).set(thing_classes=['person'])

