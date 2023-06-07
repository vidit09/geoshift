import os
import errno

from tqdm import tqdm
import pickle as pkl
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

kitti_class_name = ['car']

def get_annotation(root, image_id, ind):
    annotation_file = os.path.join(root, "Annotations", "%s.xml" % image_id)
    et = ET.parse(annotation_file)

    objects = et.findall("object")
    w = int( et.find('size').find('width').text)
    h = int( et.find('size').find('height').text)

    record = {}
    record["file_name"] = os.path.join(root, "JPEGImages", "%s.png" % image_id)
    record["image_id"] = ind
    record["annotations"] = []

    for obj in objects:
        class_name = obj.find('name').text.lower().strip()
        if class_name not in kitti_class_name:
            continue
        bbox = obj.find('bndbox')
        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = max(0,float(bbox.find('xmin').text) - 1) # fixing when -1 in anno
        y1 = max(0,float(bbox.find('ymin').text) - 1) # fixing when -1 in anno
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        box = [x1, y1, x2, y2]
        record_obj = {
        "bbox": box,
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": 0,
        }
        record["annotations"].append(record_obj)


    if len(record["annotations"]):
        record["height"] = h
        record["width"] = w
        return record
    else:
        return None


def KITTIforDetectron(root,split):

    cache_dir = os.path.join(root, 'cache')

    pkl_filename = os.path.basename(root)+'.pkl'
    pkl_path = os.path.join(cache_dir,pkl_filename)

    if os.path.exists(pkl_path):
        with open(pkl_path,'rb') as f:
            return pkl.load(f)
    else:
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            pass    

    dataset_dicts = []
    image_sets_file = os.path.join( root, "ImageSets", "Main", "%s.txt" % split)

    with open(image_sets_file) as f:
        count = 0

        for line in tqdm(f):
            record = get_annotation(root,line.rstrip(),count)
 
            if record is not None:
                dataset_dicts.append(record)
                count +=1 

    with open(pkl_path, 'wb') as f:
        pkl.dump(dataset_dicts,f)
    return dataset_dicts

def KITTIforDetectronSPLITS(root,split,splitind):

    cache_dir = os.path.join(root, 'cache')

    pkl_path = os.path.join(cache_dir,split+'.pkl')

    if os.path.exists(pkl_path):
        print('Loading data from location:',pkl_path)
        with open(pkl_path,'rb') as f:
            return pkl.load(f)
    else:
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            pass    

    dataset_dicts = []

    for ind,line in tqdm(enumerate(splitind)):
        record = get_annotation(root,line.rstrip(),ind)

        if record is not None:
            dataset_dicts.append(record)
    print('Saving pickled file at:',pkl_path)
    with open(pkl_path, 'wb') as f:
        pkl.dump(dataset_dicts,f)
    return dataset_dicts


def register_dataset(datasets_root):
    dataset_list = ['kitti_car', 
                    ]
    
    for d in dataset_list:
        try:
            cache_dir = os.path.join(datasets_root,d,'cache')
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            pass

    settype = ['train','val']
    splits = [1,2,3]
    split_inds = {}
    for d in dataset_list:
        image_sets_file = os.path.join( datasets_root,d, "ImageSets", "Main", "%s.txt" % 'train_caronly')
        all_data = []
        with open(image_sets_file) as f:
            for line in f:
                all_data.append(line.strip())

        for split in splits: 
            inds = np.arange(len(all_data))

            np.random.shuffle(inds)
            train = [all_data[i] for i in inds[:-1000]]
            val = [all_data[i] for i in inds[-1000:]]       
            for t in settype:
            
                pkl_path = os.path.join(datasets_root,d, 'cache',f'{t}_{split}_inds.pkl')
                splitname = f'{d}_{t}_{split}'
                if os.path.exists(pkl_path):
                    print('Loading inds from:', pkl_path)
                    with open(pkl_path,'rb') as f:
                        split_inds[splitname] = pkl.load(f) 
                    # import pdb;pdb.set_trace()
                else:
                    with open(pkl_path,'wb') as f:
                        split_inds[splitname] = eval(t)
                        pkl.dump(split_inds[splitname],f)
    
    allsplitsname = [t+'_'+str(s)  for t in settype for s in splits]
    for name in dataset_list:
        for ind, d in enumerate(['train_caronly']+allsplitsname):
        
            if ind == 0:
                DatasetCatalog.register(name+"_" + d, lambda datasets_root=datasets_root,name=name,d=d: KITTIforDetectron(os.path.join(datasets_root,name), d))
                MetadataCatalog.get(name+ "_" + d).set(thing_classes=['car'])
            else:
                sind = split_inds[name+'_'+d]
                DatasetCatalog.register(name+"_" + d, lambda datasets_root=datasets_root,name=name,d=d, \
                    sind=sind: KITTIforDetectronSPLITS(os.path.join(datasets_root,name), d, sind))
                MetadataCatalog.get(name+ "_" + d).set(thing_classes=['car'])

