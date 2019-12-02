# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Huizhou Li, based on the code of Girshick and Peng Zhou
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob,prep_noise_for_blob,mask_list_to_blob
import pdb

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  if cfg.USE_MASK is True:
    im_blob,im_noise, im_scales,mask,mask_shape = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}
    blobs['noise']=im_noise
    blobs['mask']=mask
    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      if num_classes<=2:
        gt_inds = np.where(roidb[0]['gt_classes'] != 100)[0]
      else:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      if num_classes<=2:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
      else:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
      [[im_blob.shape[1], im_blob.shape[2], im_scales[0], mask_shape[0][0],mask_shape[0][1]]],
      dtype=np.float32)
    return blobs
  else:
    im_blob, im_noise, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}
    blobs['noise'] = im_noise

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      if num_classes <= 2:
        gt_inds = np.where(roidb[0]['gt_classes'] != 100)[0]
      else:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      if num_classes <= 2:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
      else:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
      [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
      dtype=np.float32)
    return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  processed_mask=[]
  processed_noise = []
  im_scales = []
  mask_shapes=[]
  if cfg.USE_MASK is True:
    for i in range(num_images):
      im = cv2.imread(roidb[i]['image'])
      mask=cv2.imread(roidb[i]['mask'])
      mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      ret, mask=cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
      mask_shape = im.shape[0:2]
      mask=np.expand_dims(mask, 2)
      if roidb[i]['flipped']:
        im = im[:, ::-1, :]
        mask=mask[:, ::-1, :]
      if roidb[i]['noised']:
        row,col,ch = im.shape
        for bb in roidb[i]['boxes']:
          bcol = bb[2]-bb[0]
          brow = bb[3]-bb[1]
          mean = 0
          var = 5
          sigma = var**0.5
          gauss = np.random.normal(mean,sigma,(brow,bcol,ch))
          gauss = gauss.reshape(brow,bcol,ch)
          im = im.astype(np.float32, copy=False)
          im[bb[1]:bb[3],bb[0]:bb[2],:]=im[bb[1]:bb[3],bb[0]:bb[2],:]+gauss

      if roidb[i]['JPGed']:
        for bb in roidb[i]['boxes']:
          cv2.imwrite('JPGed.jpg',im[bb[1]:bb[3],bb[0]:bb[2],:],[cv2.IMWRITE_JPEG_QUALITY, 70])
          bb_jpged=cv2.imread('JPGed.jpg')
          im[bb[1]:bb[3],bb[0]:bb[2],:]=bb_jpged

      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im, im_scale, mask = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                      cfg.TRAIN.MAX_SIZE,mask)
      mask = np.expand_dims(mask, 2)
      im_scales.append(im_scale)
      mask_shapes.append(mask_shape)
      processed_ims.append(im)
      processed_mask.append(mask)
      noise, im_scale = prep_noise_for_blob(im, cfg.PIXEL_MEANS, target_size,
                      cfg.TRAIN.MAX_SIZE)
      processed_noise.append(noise)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    noise_blob = im_list_to_blob(processed_noise)
    mask_blob=mask_list_to_blob(processed_mask)
    return blob,noise_blob, im_scales,mask_blob,mask_shapes
  else:
    for i in range(num_images):
      im = cv2.imread(roidb[i]['image'])
      if roidb[i]['flipped']:
        im = im[:, ::-1, :]

      if roidb[i]['noised']:
        row, col, ch = im.shape
        for bb in roidb[i]['boxes']:
          bcol = bb[2] - bb[0]
          brow = bb[3] - bb[1]
          mean = 0
          var = 5
          sigma = var ** 0.5
          gauss = np.random.normal(mean, sigma, (brow, bcol, ch))
          gauss = gauss.reshape(brow, bcol, ch)
          im = im.astype(np.float32, copy=False)
          im[bb[1]:bb[3], bb[0]:bb[2], :] = im[bb[1]:bb[3], bb[0]:bb[2], :] + gauss

      if roidb[i]['JPGed']:
        for bb in roidb[i]['boxes']:
          cv2.imwrite('JPGed.jpg', im[bb[1]:bb[3], bb[0]:bb[2], :], [cv2.IMWRITE_JPEG_QUALITY, 70])
          bb_jpged = cv2.imread('JPGed.jpg')
          im[bb[1]:bb[3], bb[0]:bb[2], :] = bb_jpged

      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im, im_scale, _ = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                            cfg.TRAIN.MAX_SIZE)
      # print(mask.shape)

      im_scales.append(im_scale)

      processed_ims.append(im)

      noise, im_scale = prep_noise_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                            cfg.TRAIN.MAX_SIZE)
      processed_noise.append(noise)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    noise_blob = im_list_to_blob(processed_noise)

    return blob, noise_blob, im_scales


