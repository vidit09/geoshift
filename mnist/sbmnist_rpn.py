# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn.functional as F

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead
from detectron2.structures import ImageList
from typing import  List

@RPN_HEAD_REGISTRY.register()
class InvSTNHead(StandardRPNHead):
    def forward(self, features: List[torch.Tensor], mat: torch.Tensor):
        """
        Args:
            features (list[Tensor]): list of feature maps
        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class STNRPN(RPN):

    def forward(
        self,
        images,
        features,
        gt_instances= None,
        mat=None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        
        if mat is not None:
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features, mat)
        else:
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features, mat)


        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]


        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            #assert gt_instances is not None, "RPN requires gt_instances in training!"
            if gt_instances is not None:
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
            else:
                losses = {}
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class SBRPN(RPN):

    def forward(
        self,
        images,
        features,
        gt_instances= None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits= [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]


        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            #assert gt_instances is not None, "RPN requires gt_instances in training!"
            if gt_instances is not None:
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
            else:
                losses = {}
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        # if self.training:

        # if gt_instances is None:
        out = [
                # (N, Hi*Wi*A) -> (N, Hi, Wi, A)
                score.reshape(features[ind].shape[0],features[ind].shape[-2],features[ind].shape[-1],-1)
                for ind, score in enumerate(pred_objectness_logits)
            ]
        # else:
            # b,_,h,w = features[0].shape
            # out = [1.*(torch.stack(gt_labels)==1).reshape(b,h,w,-1)]
        return out, proposals, losses
        # else:
        #     return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class COMBRPN(RPN):

    def forward(
        self,
        images,
        features,
        gt_instances= None,
        mask = None
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        # pred_objectness_logits= [
        #     # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        #     score.permute(0, 2, 3, 1).flatten(1)
        #     for score in pred_objectness_logits
        # ]

        for ind,score in enumerate(pred_objectness_logits):
            if mask is not None:
                pred_objectness_logits[ind] = (score.permute(0,2,3,1)*mask+((mask-1)*300)).flatten(1)
            else:
                pred_objectness_logits[ind] = score.permute(0, 2, 3, 1).flatten(1)

        # pred_anchor_deltas = [
        #     # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        #     x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
        #     .permute(0, 3, 4, 1, 2)
        #     .flatten(1, -2)
        #     for x in pred_anchor_deltas
        # ]

        for ind,x in enumerate(pred_anchor_deltas):
            if mask is not None:
                pred_anchor_deltas[ind] =  (x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])\
                    .permute(0, 3, 4, 1, 2)*(mask.unsqueeze(-1))).flatten(1,-2)
            else:
                pred_anchor_deltas[ind] = x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])\
                    .permute(0, 3, 4, 1, 2).flatten(1, -2)

        if self.training:
            #assert gt_instances is not None, "RPN requires gt_instances in training!"
            if gt_instances is not None:
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
            else:
                losses = {}
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        # if self.training:

        if gt_instances is None:
            out = [
                # (N, Hi*Wi*A) -> (N, Hi, Wi, A)
                score.reshape(features[ind].shape[0],features[ind].shape[-2],features[ind].shape[-1],-1)
                for ind, score in enumerate(pred_objectness_logits)
            ]
        else:
            b,_,h,w = features[0].shape
            out = [1.*(torch.stack(gt_labels)==1).reshape(b,h,w,-1)]
        return out, proposals, losses
        # else:
        #     return proposals, losses


