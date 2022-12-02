
# GAMMA : Generative Augmentation for Attentive Marine Debris Detection

This repository is the official implementation of our work titled, "GAMMA : Generative Augmentation for Attentive Marine Debris Detection" 

## Requirements

Please follow this repository for setup, https://github.com/jwyang/faster-rcnn.pytorch

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

Download the dataset from, [drive](https://drive.google.com/file/d/1QnVqF-S4kzd9RfMj5Urbjwq0l1SJSn80/view?usp=sharing)

Put it in the main directory structure under _Pytorch_training_workspace_ directory.

## Prerequisites

Python 2.7 or 3.6
Pytorch 0.4.0
CUDA 8.0 or higher

>📋  Create a new virtual environment in anaconda and then run requirements.txt

## Training

To train the model(s) in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset marine_debris --net vgg16 --bs 4 --cuda
```

>📋  We trained on batch size of 4 on one GTX 1060 gpu. --dataset option is set to 'marine_debris' for our proposed dataset. To use resnet instead of vgg16 as backbone simply set --net option to res101. Batch size and num workers can be set as per the GPUs you are using. Other hyperparamter details are as stated in our paper. 

## Evaluation

To evaluate my model, run:

```eval
python test_net.py --dataset marine_debris --net vgg16 --checksession 1 --checkepoch 1 --checkpoint 1 --cuda
```

>📋  Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=20, CHECKPOINT=20 to test on.

## Pre-trained Models

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

- VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)
- ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

>📋  Download them and put them into the data/pretrained_model/. 
>
>NOTE. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results. If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.

## Results

Our model achieves the following state-of-the-art performance on Marine Debris Detection on our proposed dataset :

| Method         | Dataset  | Plastic |  Rov | Bio  | mAP
| -------------- |--------- | ------- |------| -----| ----
| GAMMA (Ours)   | GAMMA    | 95.6    | 90.3 | 93.0 | 93.0


