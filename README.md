# Constrained R-CNN： A general image manipulation detection model
The official repo for the paper "Constrained R-CNN Learning Rich Features for Image Manipulation Detection" 

#Overview
**Constrained R-CNN** is an end-to-end image manipulation detection model, which takes a manipulated image
as input, and predicts manipulation techniques class and pixel-level manipulation 
localization simultaneously. Compared with state-of-the-art methods, Constrained R-CNN achieves
better performance in both manipulation classification and localization. Constrained R-CNN has the 
following advantages/characteristics:

1. **Learnable:** Constrained R-CNN is not utilized any hand-drafted or predetermined features, such as SRM filter, 
frequency domain characteristics, etc.
2. **Integrity:** Constrained R-CNN performs both manipulation classification and localization, two goals of image forensics
in the real world.
3. **General** Constrianed R-CNN can detect multiple manipulation techniques, including splice, copy-move, and removal.
4. **Accuracy** Constrianed R-CNN exhibits significant improvements for manipulation localization on multiple datasets.


Constrained R-CNN mainly contains three parts as shown in following figure:

![Constrained R-CNN](https://github.com/HuizhouLi/Constrained-R-CNN/blob/master/tools/architecture.png)
1. **Learnable Manipulation Feature Extractor (LMFE)** The learnable manipulation feature extractor (LMFE) utilizes a constrained convolutional layer to 
adaptively capture feature manipulation clues ,and then the ResNet-101 learns a unified feature representation of 
multiple manipulation techniques. 


2. **Coarse Manipulation Detection (Stage-1)** In Stage-1, the attention region proposal network effectively discriminates
 manipulated regions. The prediction module performs the manipulation classification and coarse localization on bounding box level.
 
 
3. **Fine Manipulation Detection (Stage-2)** In Stag-2, the skip structure fuses low-level and high-level information to
 refine the global manipulation features. Finally, the coarse localization information guides the model to further learn
  the finer local features and segment out the tampered region.




# Dependencies
Constrained R-CNN is written in Tensorflow. Some packages need to be installed.
  
  - Python  3.6.7, 
  
  - CUDA   8.0.44 
  
  - CUDNN  5.1

Other packages please run requirements.txt:
```
pip install -r requirements.txt
```

# Installation:
Build the Cython modules, run the command:
```
cd lib
make clean
make
cd ..
```

For more details, see:
 - Faster R-CNN: https://github.com/endernewton/tf-faster-rcnn/
 - RGB-N: https://github.com/pengzhou1108/RGB-N

# Directory
```.
├── cfgs                                                            
├── data                                                             # Model Weights
│   ├── CASIA_weights                                                # Trained on CASIA-2 dataset
│   ├── COVER_weights                                                # Trained on Coverage dataset
│   ├── imagenet_weights                                             # Imagenet weights(ResNet-101)
│   ├── ini_weights_old                                              # Trained on COCO Synthetic dataset
│   └── NIST_weights                                                 # Trained on NIST16 dataset
├── Data_preprocessing                                               # Data pre-process script
├── dataset                                                          # Dateset directory
│   ├── CASIA
│   ├── Columbia
│   ├── COVER_DATASET
│   └── NIST2016
├── lib                                                              # Model
├── test_image
├── tools
│   ├── demo.py
│   ├── test_net.py
│   └── trainval_net.py
├── train_faster_rcnn.sh
├── requirements.txt
├── test_faster_rcnn.sh
├── Demo.ipynb                                                       # Demo
```
#Model weights
We provide model weights trained on multiple dataset. You can download here, and put them into "data" folder as shown in directory tree.
# Data Pre-processing
Constrained R-CNN needs manipulation techniques class , bounding box coordinates and ground truth mask for training. We extract
the bounding box from the ground truth mask. For different datasets, the pre-processing is slightly different:<br>

 
  Dateset | Class Label |GT Mask  | Pre-process  |
  :--- |:---:|:---:|---|
  |CASIA|Authentic, Tamper|N | 1. Generate groundtruth mask<br>2. Extract bounding box<br>3. Split training and testing set|
  |COVER|Authentic, Tamper|Y |1. Extract bounding box from mask<br>2. Split training and testing set|
  |Columbia|Authentic, Tamper|Y |1. Extract bounding box from mask<br>2. All images for testing|
  |NIST2016|Splice, Copy-move, Removal|Y|1. Extract bounding box from mask<br>2. Split training and testing set|
  
For more details, please see the scripts in the `Data preprocessing` folder:

```
CAISA1_preprocess.ipynb
Columbia_preprocess.ipynb
COVER_preprocess.ipynb
NIST_preprocess.ipynb
```

  
#Demo
If you only want to test Constrained R-CNN simply, please download the repo and play 
with the provided ipython notebook.
```
Demo.ipynb
```
#Train
If you want to retrain the Constrained R-CNN, please follow this process:<br>
##Pre-trained model (Only Stage-1)
 We also provide the pre-trained weight(Only Stage-1) in `/data/ini_weight_old/`.<br><br>
 1.Download COCO synthetic dataset (https://github.com/pengzhou1108/RGB-N)<br>
 2.Change the coco synthetic path in `lib/datasets/factory.py`:
```
coco_path= #FIXME
for split in ['coco_train_filter_single', 'coco_test_filter_single']:
    name = split
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))
```

 3.Specify the ImageNet resnet101 model path in `train_faster_rcnn.sh` as below:

```
        python3 ./tools/trainval_net.py \
            --weight ./imagenet_weights/res101.ckpt \ #ImageNet resnet101 model path
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
```
 4.Specify the dataset, gpu, and network in `train_faster_rcnn.sh` as below as run the file
```
./train_faster_rcnn.sh 0 coco res101_cbam EXP_DIR coco_flip_cbam
```
 5.Testing pre-trained model
```
./test_faster_rcnn.sh 0 coco res101_cbam EXP_DIR coco_flip_cbam
```
##Trained on Standard Datasets (Entire model)
The weights we have completed for training are stored in `data`. 
If you want to retrain Constrained R-CNN, please follow the process:
1. Data pre-process as before mentioned.
2. Change the dataset path in `lib/datasets/factory.py`, such as:
```
nist_path='../dataset/NIST2016'    #dataset path
for split in ['dist_NIST_train_new_2', 'dist_NIST_test_new_2']:
    name = split
    __sets[name] = (lambda split=split: nist(split,2007,nist_path))
```
 3.Specify the pre-trained model path in `train_faster_rcnn.sh` as below:

```
        python3 ./tools/trainval_net.py \
            --weight ./ini_weights_old/res101_cbam_faster_rcnn_iter_110000.ckpt \ #COCO synthetic dataset pre-trained model path
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
```
 4.Specify the dataset, gpu, and network in `train_faster_rcnn.sh` as below as run the file
```
./train_faster_rcnn.sh 0 nist res101_C3-R-cbam EXP_DIR NIST_flip_C3RCBAM
```

# Test the model
1. Check the model path match well with `NET_FINAL` in `test_faster_rcnn.sh`, making sure the checkpoint iteration exist in model output path. Otherwise, change the iteration number `ITERS` as needed.
```
  NIST)
    TRAIN_IMDB="dist_NIST_train_new_2"
    TEST_IMDB="dist_NIST_test_new_2"
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
```

2. Run `test_dist_faster.sh`. If things go correcty, it should print out `F1 score` and `AUC` indicating the detection result.

