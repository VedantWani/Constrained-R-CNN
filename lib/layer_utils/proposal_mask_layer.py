# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Huizhou Li
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms
import cv2


def proposal_mask_layer(rois, cls_prob, bbox_pred, im_info,num_classes,training,testing):

  image_info=im_info[0]

  boxes = rois[:, 1:5]/image_info[2]

  scores = np.reshape(cls_prob, [cls_prob.shape[0], -1])

  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

  stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes))
  means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes))
  bbox_pred *= stds
  bbox_pred += means

  box_deltas = bbox_pred
  pred_boxes = bbox_transform_inv(boxes, box_deltas)

  pred_boxes = _clip_boxes(pred_boxes, image_info[3:5])


  mask_data_list=[]
  for ind in range(1,num_classes):
    if ind==0:
      continue
    else:
      cls_boxes = pred_boxes[:, 4 * ind:4 * (ind + 1)]
      cls_boxes = cls_boxes*image_info[2]
      cls_scores = scores[:, ind]
      dets = np.hstack((cls_boxes,
                        cls_scores[:, np.newaxis])).astype(np.float32)

      if training==1 and testing ==0:
        keep = nms(dets, 0.7)
        dets = dets[keep, :]
      cls_ind = np.full((dets.shape[0]), ind,dtype=np.float32)
      batch_inds = np.zeros((dets.shape[0]), dtype=np.float32)

      dets = np.hstack((batch_inds[:,np.newaxis], dets, cls_ind[:,np.newaxis]))


      mask_data_list.extend(dets.tolist())

  if len(mask_data_list) :
    if training==1 and testing==0:
      mask_batch=cfg.TRAIN.MASK_BATCH
    elif training==0 and testing==1:
      mask_batch = cfg.TEST.MASK_BATCH

    mask_data=np.array(mask_data_list,dtype=np.float32)

    mask_data=mask_data[np.argsort(mask_data[:,5])[::-1]]

    if mask_data.shape[0]>mask_batch:
      mask_data=mask_data[0:mask_batch,:]

  else:
    mask_data=None
  return mask_data
def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes