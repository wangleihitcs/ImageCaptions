### ImageCaptions
a base model for image captioning

### Config
- python 2.7
- tensorflow 1.8.0
- python package 
    * nltk
    * PIL
    * json
    * numpy

### DataDownload
- coco image dataset
    * you need to download [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
    * then unzip it to dir 'data/train2017/'
- coco image annotations
    * you need to download [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
    * then unzip it:
        * copy 'captions_train2017.json' to dir 'data/coco_annotations'
- pretrain inception model
    * you need to download [inception_v3.ckpt](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
) to dir 'data/inception/'

## Train
#### First, get post proccess data
- get 'data/captions.json', 'data/captions_gt.json'
    ```shell
    cd preproccess
    python data_entry.py    
    ```
- get 'data/train_split.json', 'data/val_split.json', 'data/test_split.json'
    ```shell
    cd preproccess
    python split.py    
    ```
- get 'data/vocabulary.json'
    ```shell
    cd preproccess
    python vocabulary.py    
    ```
#### Second, get TFRecord files
Because dataset is too large, we should do some operations to purse speed and CPU.GPU efficiency.
You need to wait 30 mins to convert data to 'data/train/train-xx.tfrecord', I convert Train Data to 40 tfrecord files.
* get 'data/train/train-00.tfrecord' - 'data/train/train-39.tfrecord'
    ```shell
        python datasets.py    
    ```
* so you need get 'data/tfrecord_list.json' for tensorflow filename queue, it is easy
    
#### Third, let's go train
```shell
        python main.py    
```

## Test

## Result


## Summary
The model is very very*N simple, I never adjust the hyperparameter, so if you want, you could do that.
