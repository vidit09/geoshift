import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer


from .stn_arch import build_stn_arch

    
@META_ARCH_REGISTRY.register()
class STNPerspectiveRCNN(GeneralizedRCNN):
    
    def __init__(self,cfg):
        super().__init__(cfg)
        self.stn = build_stn_arch(cfg)
        # nstn = len(self.stn.affine_params)
        nstn = self.stn.nbstn

        self.aggregator = self.build_aggregator(nstn)
        self.tracking_logits = 0
        self.counter = 0
    
    
    def build_aggregator(self,nstn):
        inchannel = 1024*nstn
        module  = nn.Sequential(
            nn.Conv2d(inchannel,inchannel//2,3,padding=1),
            nn.BatchNorm2d(inchannel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel//2,inchannel//nstn,3,padding=1),
            nn.BatchNorm2d(inchannel//nstn),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel//nstn,inchannel//nstn,1,padding=1),
            nn.BatchNorm2d(inchannel//nstn),
            nn.ReLU(inplace=True),
        )
        return module

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        
        images.tensor = images.tensor*self.pixel_std+self.pixel_mean
        images = self.stn(images)

        for ind, im in enumerate(images):
            images[ind].tensor = (images[ind].tensor-self.pixel_mean)/self.pixel_std

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        else:
            gt_instances = None 


        all_features = []
        for i in range(len(images)):    
            features = self.backbone(images[i].tensor)
            for k, v in features.items():
                all_features.append(features[k])

        all_features = self.stn.inverse(all_features)

        features = {'res4': self.aggregator(torch.cat(all_features,1))}
        images = images[0]

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

            
                    tfimage = images.tensor[0].detach()
                    tfimage = tfimage*self.pixel_std + self.pixel_mean
                    tfimage = convert_image_to_rgb(tfimage.permute(1, 2, 0), self.input_format)
                    box_size = min(len(proposals[0].proposal_boxes), 20)
                    b = proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
                    v_pred = Visualizer(tfimage, None).overlay_instances(boxes=b).get_image()
                    try:
                        # shape might change after transform
                        oh,ow,c = o_pred.shape
                        vh,vw,c = v_pred.shape
                        combined_im = np.zeros((max(oh,vh),ow+vw,c),'uint8')
                        combined_im[:oh,:ow,:] = o_pred
                        combined_im[:vh,ow:,:] = v_pred
                        #vis_img = np.concatenate((o_pred, v_pred), axis=1)
                        vis_img = combined_im
                    except:
                        import pdb;pdb.set_trace()

                    vis_img = vis_img.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

            for ind, v in enumerate(self.stn.params.reshape(-1).detach().cpu().numpy()):
                        storage.put_scalar(str(ind),v)


        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        images.tensor = images.tensor*self.pixel_std+self.pixel_mean 
        images = self.stn(images)

        for ind, im in enumerate(images):
            images[ind].tensor = (images[ind].tensor-self.pixel_mean)/self.pixel_std

        all_features = []
        for i in range(len(images)):    
            features = self.backbone(images[i].tensor)
            for k, v in features.items():
                all_features.append(features[k])
        
        all_features = self.stn.inverse(all_features)
       
        # mf = [all_features[i].max(1,True)[0] for i in range(len(images))]
        # import torchvision
        # for ind,ff in enumerate(mf):
        #     ff = (ff-ff.min())/(ff.max()-ff.min())
        #     torchvision.utils.save_image(ff,f'test/scale{ind}.png')
        features = {'res4': self.aggregator(torch.cat(all_features,1))}
        
        # ff = features['res4'].max(1,True)[0]
        # ff = (ff-ff.min())/(ff.max()-ff.min())
        # torchvision.utils.save_image(ff,'test/scaleagg.png')
        # import pdb;pdb.set_trace()

        images = images[0]
        if detected_instances is None:
            if self.proposal_generator is not None:
                logits,proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        
        # if True:
        #     self.check_rpn_dis(logits)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


    def check_rpn_dis(self,logits):
        if self.counter ==0 :
            self.tracking_logits = torch.sigmoid(logits[0])
        else:
            import numpy as np
            b,h1,w1,c = logits[0].shape
            b,h2,w2,c = self.tracking_logits.shape
            # print(h1,w1)
            # print(h2,w2)
            maxh = max(h1,h2)
            maxw = max(w1,w2)

            padh = (maxh - h1)
            padw = (maxw - w1)
            if padh == 1 and padw == 1:
                nlogt = F.pad(torch.sigmoid(logits[0])>0.5,(0,0,0,1,0,1),'constant',0)
            elif padh == 1 :
                nlogt = F.pad(torch.sigmoid(logits[0])>0.5,(0,0,int(np.floor(padw/2)) ,int(np.ceil(padw/2)),0,1),'constant',0)
            elif padw == 1:
                nlogt = F.pad(torch.sigmoid(logits[0])>0.5,(0,0,0,1,int(np.floor(padh/2)),int(np.ceil(padh/2))),'constant',0)
            else:
                nlogt = F.pad(torch.sigmoid(logits[0])>0.5,(0,0,int(np.floor(padw/2)),int(np.ceil(padw/2)),int(np.floor(padh/2)),int(np.ceil(padh/2))),'constant',0)

            padh = (maxh - h2)
            padw = (maxw - w2)
            
            if padh == 1 and padw == 1:
                self.tracking_logits = F.pad(self.tracking_logits,(0,0,0,1,0,1),'constant',0)
            elif padh == 1 :
                self.tracking_logits = F.pad(self.tracking_logits,(0,0,int(np.floor(padw/2)),int(np.ceil(padw/2)),0,1),'constant',0)
            elif padw == 1:
                self.tracking_logits = F.pad(self.tracking_logits,(0,0,0,1,int(np.floor(padh/2)),int(np.ceil(padh/2))),'constant',0)
            else:
                self.tracking_logits = F.pad(self.tracking_logits,(0,0,int(np.floor(padw/2)),int(np.ceil(padw/2)),int(np.floor(padh/2)),int(np.ceil(padh/2))),'constant',0)

            self.tracking_logits = self.tracking_logits + nlogt
        self.counter += 1
        if self.counter == 2692: #498: #2692:
            import pdb;pdb.set_trace()  

