# --------------------------------------------------------
# Tensorflow RGB-N
# Licensed under The MIT License [see LICENSE for details]
# Written by Huizhou Li, based on the code of Peng Zhou
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.coco import coco
from datasets.casia import casia
from datasets.cover import dist_fake
from datasets.nist import nist
from datasets.columbia import dvmm


columbia_path='/media/li/Li/Columbia'
for split in ['dist_train_all_single', 'dist_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: dvmm(split,2007,columbia_path))

cover_path='/media/li/Li/cover/'
for split in ['dist_cover_train_single', 'dist_cover_test_single']:
    name = split
    __sets[name] = (lambda split=split: dist_fake(split,2007,cover_path))

nist_path='/media/li/Li/NIST2016'
for split in ['dist_NIST_train_new_2', 'dist_NIST_test_new_2']:
    name = split
    __sets[name] = (lambda split=split: nist(split,2007,nist_path))

casia_path='/media/li/Data/CASIA'
#for split in ['casia_train_all_single', 'casia_test_all_1']:
for split in ['casia_train_all_single', 'casia_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: casia(split,2007,casia_path))

coco_path='/media/li/Li/filter_tamper'
for split in ['coco_train_filter_single', 'coco_test_filter_single']:
    name = split
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
