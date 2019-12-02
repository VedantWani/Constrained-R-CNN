# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Huizhou Li
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob
from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    row, col, ch = im.shape
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)

    im_orig = im.astype(np.float32, copy=True)
    # im_orig = im_orig + gauss
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    processed_noise = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        noise = im
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        processed_noise.append(noise)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    noise_blob = im_list_to_blob(processed_noise)
    return blob, noise_blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], blobs['noise'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


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


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)

    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']

    # seems to have height, width, and image scales
    # still not sure about the scale, maybe full image it is 1.

    if cfg.USE_MASK is True:
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0], im.shape[0], im.shape[1]]],
                                    dtype=np.float32)
        scores1, scores, bbox_pred, rois, feat, s, y_preds, mask_data,layers = net.test_image(sess, blobs['data'], blobs['noise'],
                                                                                       blobs['im_info'])
        
        # Found and solved some bugs, I need a drink.
        boxes = rois[:, 1:5] / im_scales[0]
        mask_boxes = mask_data[:, 1:5]/im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
        mask_scores = np.reshape(mask_data[:, 5], [mask_data[:, 5].shape[0], -1])
        mask_boxes = np.reshape(mask_boxes, [mask_boxes.shape[0], -1])
        maskcls_ind = np.reshape(mask_data[:, -1], [mask_data[:, -1].shape[0], -1])

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(boxes, box_deltas)

            pred_boxes = _clip_boxes(pred_boxes, im.shape)
            pred_mask_boxes = mask_boxes
            # pred_boxes = _clip_boxes(boxes, im.shape)

        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            pred_mask_boxes = np.tile(mask_boxes, (1, mask_scores.shape[1]))
        return scores, pred_boxes, feat, s, maskcls_ind, pred_mask_boxes, mask_scores, y_preds, mask_data, layers
    else:
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        scores1, scores, bbox_pred, rois, feat, s = net.test_image(sess, blobs['data'], blobs['noise'],
                                                                   blobs['im_info'])

        boxes = rois[:, 1:5] / im_scales[0]

        # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
        scores = np.reshape(scores, [scores.shape[0], -1])

        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(boxes, box_deltas)

            pred_boxes = _clip_boxes(pred_boxes, im.shape)


        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        return scores, pred_boxes, feat, s


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    # assert prediction.dtype == np.uint8
    # assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    y_test=gt.flatten()
    y_pred=prediction.flatten()
    precision,recall,thresholds=metrics.precision_recall_curve(y_test,y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    return precision, recall,auc_score


def cal_fmeasure(precision, recall):

    max_fmeasure = max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])
    return max_fmeasure


def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    all_f1 = np.zeros((imdb.num_images, imdb.num_classes), np.float)
    all_auc = np.zeros((imdb.num_images, imdb.num_classes), np.float)
    all_auc_new = np.zeros((imdb.num_images, imdb.num_classes), np.float)
    counters = []
    output_dir = get_output_dir(imdb, weights_filename)
    if os.path.isfile(os.path.join(output_dir, 'detections.pkl')):
        all_boxes = pickle.load(open(os.path.join(output_dir, 'detections.pkl'), 'r'))

    else:
        # timers
        if cfg.USE_MASK is True:
            _t = {'im_detect': Timer(), 'mask': Timer()}

            for i in range(num_images):
                print(imdb.image_path_at(i))
                im = cv2.imread(imdb.image_path_at(i))

                mask_gt = cv2.imread(imdb.mask_path_at(i))
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                ret, mask_gt = cv2.threshold(mask_gt, 127, 255, cv2.THRESH_BINARY)
                mask_gt = (mask_gt / 255.0).astype(np.float32)

                _t['im_detect'].tic()

                scores, boxes, feat, s, maskcls_inds, mask_boxes, mask_scores, mask_pred, _ ,_= im_detect(sess, net, im)

                _t['im_detect'].toc()

                _t['mask'].tic()

                # skip j = 0, because it's the background class
                # for j in range(1, imdb.num_classes):
                #     inds = np.where(scores[:, j] > thresh)[0]
                #     cls_scores = scores[inds, j]
                #     cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                #     cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                #         .astype(np.float32, copy=False)
                #     keep = nms(cls_dets, cfg.TEST.NMS)
                #     cls_dets = cls_dets[keep, :]
                #     all_boxes[j][i] = cls_dets

                batch_ind = np.where(mask_scores > 0.)[0]
                cls=maskcls_inds[np.argmax(mask_scores)].astype(int)
                mask_boxes=mask_boxes.astype(int)
                if batch_ind.shape[0] == 0:
                    f1 = 1e-10
                    auc_score = 1e-10
                else:
                    mask_out = np.zeros(im.shape[:2],dtype=np.float)
                    for ind in batch_ind:
                        height = mask_boxes[ind, 3] - mask_boxes[ind, 1]
                        width = mask_boxes[ind, 2] - mask_boxes[ind, 0]
                        if width <= 0 or height <= 0:
                            continue
                        else:
                            mask_box_pre = cv2.resize(mask_pred[ind, :, :, :], (width, height))
                            mask_pre = np.zeros(im.shape[:2],dtype=np.float)
                            bbox1 = mask_boxes[ind, :]
                            mask_pre[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = mask_box_pre
                            mask_out = np.where(mask_out >= mask_pre, mask_out, mask_pre)
                    precision, recall,auc_score = cal_precision_recall_mae(mask_out, mask_gt)
                    f1 = cal_fmeasure(precision, recall)
                    f1=np.max(np.array(f1))
                print('F1 score per image：',f1)
                print('AUV score per image：', auc_score)
                all_f1[i, cls] = f1
                all_auc[i, cls] = auc_score
                _t['mask'].toc()
                print('Im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                      .format(i + 1, num_images, _t['im_detect'].average_time,
                              _t['mask'].average_time))
            class_f1 = np.zeros(imdb.num_classes)
            class_auc = np.zeros(imdb.num_classes)
            for j in range(1, imdb.num_classes):
                cls_f1 = all_f1[:, j]
                f1_ind = np.where(cls_f1 > 0.)[0]
                cls_f1 = cls_f1[f1_ind]
                class_f1[j] = np.average(cls_f1)

                cls_auc = all_auc[:, j]
                auc_ind = np.where(cls_auc > 0)[0]
                cls_auc = cls_auc[auc_ind]
                class_auc[j] = np.average(cls_auc)
            # det_file = os.path.join(output_dir, 'detections_{:f}.pkl'.format(10))
            # with open(det_file, 'wb') as f:
            #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        else:
            _t = {'im_detect': Timer(), 'compute': Timer()}
            for i in range(num_images):
                im = cv2.imread(imdb.image_path_at(i))

                _t['im_detect'].tic()
                scores, boxes, _, _ = im_detect(sess, net, im)
                _t['im_detect'].toc()

                _t['compute'].tic()

                # skip j = 0, because it's the background class
                for j in range(1, imdb.num_classes):
                    inds = np.where(scores[:, j] > thresh)[0]
                    cls_scores = scores[inds, j]
                    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    all_boxes[j][i] = cls_dets

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in range(1, imdb.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, imdb.num_classes):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]
                _t['compute'].toc()

                print('Im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                      .format(i + 1, num_images, _t['im_detect'].average_time,
                              _t['compute'].average_time))

            det_file = os.path.join(output_dir, 'detections_{:f}.pkl'.format(10))
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            imdb.evaluate_detections(all_boxes, output_dir)

    if cfg.USE_MASK is True:

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Test Results:')
        print('Average F1  Score: %.3f' %np.average(class_f1[1:]))
        print('Average AUC Score: %.3f' %np.average(class_auc[1:]))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('\n')
        print('============================================================')
        print('Constrained R-CNN')
        print('A general image manipulation detection model.')
        print('Licensed under The MIT License [see LICENSE for details]')
        print('Written by Huizhou Li')
        print('============================================================')



