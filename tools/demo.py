#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Constrained R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Huizhou Li , based on code of Peng Zhou and Xinlei Chen
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from skimage import io
from utils.cython_nms import nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')

import numpy as np
import os, cv2
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from nets.resnet_v1_cbam import resnet_cbam
from nets.res101_v1_C3Rcbam import resnet_C3Rcbam
from utils.blob import im_list_to_blob
from model.bbox_transform import clip_boxes, bbox_transform_inv
import csv
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# CLASSES=('authentic', 'tamper')
CLASSES=('authentic',  # always index 0
                       'splice', 'removal', 'copy-move')
data_dir = '/media/li/Li/NIST2016/'
data_dir_2 = '/media/li/Li/NIST2016/'

data_dir = '/media/li/Li/NIST2016/probe/'

vis_dir = '/media/li/Li/lihuizhou/RGB-N/vis_result/'
NETS = {'NIST_flip_0001_C3Rcbam_new':('res101_cbam_faster_rcnn_iter_110000.ckpt',),}
DATASETS = {'NIST_train_new_2': ('dist_NIST_train_new_2',),
            'NIST_test_new_2': ('dist_NIST_test_new_2',), }
test_single = True

def cal_precision_recall_mae(prediction, gt):
    y_test=gt.flatten()
    y_pred=prediction.flatten()
    precision,recall,thresholds=metrics.precision_recall_curve(y_test,y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    return precision, recall,auc_score

def cal_fmeasure(precision, recall):

    fmeasure = [[(2 * p * r) / (p + r + 1e-10)] for p, r in zip(precision, recall)]
    fmeasure=np.array(fmeasure)
    fmeasure=fmeasure[fmeasure[:, 0].argsort()]

    max_fmeasure=fmeasure[-1,0]
    return max_fmeasure

def demo(sess, net, image_name,maskpath,classes):

    try:
        im_file = os.path.join(data_dir, image_name)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))

    imfil=im_file

    im = cv2.imread(imfil)
    save_id=str(str(imfil.split('/')[-1]).split('.')[0])
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # try:

    scores, boxes, feat, s, maskcls_inds, mask_boxes, mask_scores, mask_pred, mask_data,layers= im_detect(sess, net, im)

    timer.toc()
    noise =np.squeeze(layers['noise'])
    noise += cfg.PIXEL_MEANS
    noise_save=noise.copy()
    noise_save=cv2.resize(noise_save,(im.shape[1],im.shape[0]))
    cv2.imwrite('/media/li/Li/'+save_id+'_authentic.png', im)
    cv2.imwrite('/media/li/Li/'+save_id+'_noise.png',noise_save)
    batch_ind = np.where(mask_scores > 0.)[0]
    mask_boxes = mask_boxes.astype(int)
    if batch_ind.shape[0] == 0:
        f1 = 1e-10
        auc_score = 1e-10
    else:
        mask_out = np.zeros(im.shape[:2], dtype=np.float)
        for ind in batch_ind:
            height = mask_boxes[ind, 3] - mask_boxes[ind, 1]
            width = mask_boxes[ind, 2] - mask_boxes[ind, 0]
            if width <= 0 or height <= 0:
                continue
            else:
                mask_inbox = cv2.resize(mask_pred[ind, :, :, :], (width, height))
                mask_globe = np.zeros(im.shape[:2], dtype=np.float)
                bbox1 = mask_boxes[ind, :]
                mask_globe[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = mask_inbox
                mask_out = np.where(mask_out >= mask_globe, mask_out, mask_globe)
        cv2.imwrite('/media/li/Li/' + save_id + '_pre_mask.png', mask_out*255)
        hotmap = cv2.applyColorMap(np.uint8(255 * mask_out.copy()), cv2.COLORMAP_JET)
        cv2.imwrite('/media/li/Li/' +save_id + '_hotmap.png', hotmap)


        mask_gt=cv2.imread(maskpath)
        mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
        ret, mask_gt = cv2.threshold(mask_gt, 127, 1, cv2.THRESH_BINARY)
        mask_gt = mask_gt.astype(np.float32)
        precision, recall, auc_score = cal_precision_recall_mae(mask_out, mask_gt)
        f1 = cal_fmeasure(precision, recall)
        ret,mask_out=cv2.threshold(mask_out, 0.5, 1, cv2.THRESH_BINARY)
        img_add = cv2.addWeighted(im, 0.5, hotmap, 0.5,gamma=0,)
        mask_ind=np.where(mask_out>0)
        mask_in=im[mask_ind]
        img_add[mask_ind]=mask_in

        img_add = Image.fromarray(cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB))
        fig, ax = plt.subplots(figsize=(10,10))
        img_add=ax.imshow(img_add, aspect='equal')
        avg_score = 0
        dets = np.hstack((mask_boxes,
                          mask_scores)).astype(np.float32)
        keep = nms(dets, 0.5)
        mask_boxes=mask_boxes[keep,:]
        mask_scores=mask_scores[keep,:]
        maskcls_inds=maskcls_inds[keep,:]
        for i in range(len(mask_scores)):
            bbox = mask_boxes[i,:]
            score = mask_scores[i]
            avg_score = max(avg_score, score)
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=5)
            )
            ax.text(bbox[0], bbox[1] - 22,
                    '{:s} '.format(str(classes[int(maskcls_inds[i])])),
                    bbox=dict(facecolor='red', alpha=0.8),
                    fontsize=38, color='white')

            plt.axis('off')
            plt.draw()
        plt.savefig('{}.png'.format(
            os.path.join('/media/li/Li/' + save_id + '_result')))
        plt.close(fig)

    return f1, auc_score


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='NIST_flip_0001_C3Rcbam_new')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default= 'NIST_test_new_2')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    cfg.USE_MASK = True
    cfg.TEST.MASK_BATCH=8

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('/media/li/Data/Constrained-R-CNN/data/NIST_weights/C3_R-cbam/res101_mask_faster_rcnn_iter_60000.ckpt')


    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'NIST_flip_0001_C3Rcbam_new':
        net = resnet_C3Rcbam(batch_size=1, num_layers=101)
    elif demonet == 'NIST_flip_0001_cbam_new':
        net = resnet_cbam(batch_size=1, num_layers=101)


    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                            tag='default', anchor_scales=[8, 16, 32, 64],
                            anchor_ratios=[0.5, 1, 2])
    saver = tf.train.Saver()

    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    print('~~~~~~~~~~~~~~~start~~~~~~~~~~~~~~~~~~~~')
    score_save = []
    label = []
    save_name = []
    f1score=[]
    aucscore=[]
    if test_single:

        im_names=['/media/li/Data/Constrained-R-CNN/test_image/NC2016_7894_splice.png']

        for im_name in im_names:
            if not os.path.isfile('{}.png'.format(os.path.join(vis_dir, im_name))):
                maskpath = '/media/li/Data/Constrained-R-CNN/test_image/NC2016_7894_gt.png'
                f1, auc_score=demo(sess, net, im_name,maskpath)
                print(f1, auc_score)

    else:
        with open(os.path.join('/media/li/Li/lihuizhou/RGB-N/coco_synthetic/', 'test_filter_single.txt'), 'r') as f:
            im_names = f.readlines()
            im_ind=1
            im_num=len(im_names)
            for file in im_names:
                print(' {:d}/{:d} images'.format(im_ind, im_num))
                im_name = file.split(' ')[0].split('/')[-1]
                im_ind+=1
                # image_base=im_name[:11]
                maskpath='/media/li/Li/NIST2016/mask'
                im_box = [float(file.split(' ')[i]) for i in range(1, 5)]
                im_label=str(file.split(' ')[5])
                if not os.path.isfile('{}.png'.format(os.path.join(vis_dir, im_name))):
                    f1,auc_score = demo(sess, net, im_name,maskpath)
                    if len(f1)>1:
                        temp=np.array(f1)
                        temp_f1=max(temp)
                        auc_temp1=np.array(auc_score)
                        auc_temp=max(auc_temp1)

                    elif len(f1)==0:
                        # continue
                        temp_f1=[0.0]
                        auc_temp=[0.0]
                    else:
                        temp_f1=f1
                        auc_temp=auc_score

                    label.append(im_label)
                    save_name.append(im_name)
                    f1score.append(temp_f1[0])
                    aucscore.append(auc_temp[0])

    # temp=np.array(f1score)
    # tempauc=np.array(aucscore)
    #
    # mean_f1=np.mean(temp)
    # mean_auc=np.mean(tempauc)
    # print('F1 score is ：  ',mean_f1)
    # print('AUC score is ：  ', mean_auc)
    print('~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~')
    # plt.show()
