# -*- coding: UTF-8 -*-
import tensorflow as tf
import json
import nltk
import os
import numpy as np
from utils import image_utils

def get_train_batch(tfrecord_list_path, batch_size, image_size=224, max_caption_length=25):
    with open(tfrecord_list_path, 'r') as file:
        tfrecord_path_list = json.load(file)
    # 1. get filename_queue
    filename_queue = tf.train.string_input_producer(tfrecord_path_list, shuffle=False)

    # 2. get image pixels, sentence, mask, image_id
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image_pixels': tf.FixedLenFeature([image_size*image_size*3], tf.float32),
            'sentence': tf.FixedLenFeature([max_caption_length], tf.int64),
            'mask': tf.FixedLenFeature([max_caption_length], tf.int64),
            'image_id': tf.FixedLenFeature([1], tf.int64),
        }
    )
    image = tf.reshape(features['image_pixels'], [image_size, image_size, 3])
    sentence = features['sentence']
    mask = features['mask']
    image_id = features['image_id']

    # 3. get tf.train.batch
    min_after_dequeue = 10000
    image_batch, sentece_batch, mask_batch, image_id_batch = tf.train.batch(
        [image, sentence, mask, image_id],
        batch_size=batch_size,
        capacity=min_after_dequeue + 3*batch_size
    )

    return image_batch, sentece_batch, mask_batch, image_id_batch

# get data from memory
def get_inputs(split_list_path, image_size=224, max_caption_length=25):
    imgs_path = '/home/wanglei/workshop/MSCoCo/train2017'
    captions_path = 'data/captions.json'
    vocabulary_path = 'data/vocabulary.json'

    with open(vocabulary_path, 'r') as file:
        vocabulary = json.load(file)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i
    with open(captions_path, 'r') as file:
        caption_dict = json.load(file)
    with open(split_list_path, 'r') as file:
        split_id_list = json.load(file)

    image_list, sentence_list, mask_list, image_id_list = [], [], [], []
    for image_id in split_id_list:
        filename = str('%012d' % image_id) + '.jpg'
        img_file_path = os.path.join(imgs_path, filename)
        image = image_utils.getImages(img_file_path, image_size)
        image_list.append(image)

        caption = caption_dict[str(image_id)][0]
        word_list = nltk.word_tokenize(caption)
        if word_list.__len__() < max_caption_length:
            word_list.append('</S>')
            for _ in range(max_caption_length - word_list.__len__()):
                word_list.append('<EOS>')
        else:
            word_list = word_list[:max_caption_length]
        sentence = np.zeros(shape=[max_caption_length])
        mask = np.ones(shape=[max_caption_length])
        for i in range(max_caption_length):
            sentence[i] = word2id[word_list[i]]
            if word_list[i] == '<EOS>':
                mask[i] = 0
        sentence_list.append(sentence)
        mask_list.append(mask)

        image_id_list.append([image_id])

    return image_list, sentence_list, mask_list, image_id_list

# get data/train/xxx.tfrecord
def get_train_tfrecord(imgs_path, captions_path, split_list_path, vocabulary_path, image_size=224, max_caption_length=25):
    vocabulary = get_vocabulary(vocabulary_path)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i
    with open(captions_path, 'r') as file:
        caption_dict = json.load(file)
    with open(split_list_path, 'r') as file:
        split_id_list = json.load(file)

    D = 40
    for i in range(D):
        subsets_num = split_id_list.__len__() / D + 1
        sub_split_id_list = split_id_list[i * subsets_num: (i + 1) * subsets_num]

        train_tfrecord_name = 'data/train/train-%02d.tfrecord' % i
        writer = tf.python_io.TFRecordWriter(train_tfrecord_name)

        for image_id in sub_split_id_list:
            filename = str('%012d' % image_id) + '.jpg'
            img_file_path = os.path.join(imgs_path, filename)
            image = image_utils.getImages(img_file_path, image_size)
            image = image.reshape([image_size*image_size*3])

            caption = caption_dict[str(image_id)][0]
            word_list = nltk.word_tokenize(caption)
            if word_list.__len__() < max_caption_length:
                word_list.append('</S>')
                for _ in range(max_caption_length - word_list.__len__()):
                    word_list.append('<EOS>')
            else:
                word_list = word_list[:max_caption_length]
            sentence = np.zeros(shape=[max_caption_length], dtype=np.int64)

            mask = np.ones(shape=[max_caption_length], dtype=np.int64)

            for i in range(max_caption_length):
                sentence[i] = word2id[word_list[i]]
                if word_list[i] == '<EOS>':
                    mask[i] = 0

            # image_raw = image.tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'image_pixels': tf.train.Feature(float_list = tf.train.FloatList(value = image)),
                        'sentence': tf.train.Feature(int64_list = tf.train.Int64List(value = sentence)),
                        'mask': tf.train.Feature(int64_list = tf.train.Int64List(value = mask)),
                        'image_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_id]))
                    }
                )
            )
            serialized = example.SerializeToString()
            writer.write(serialized)
            # print('%s' % filename)
        print('%s write to tfrecord success!' % train_tfrecord_name)
        # break

def main():
    imgs_dir_path = '/home/wanglei/workshop/MSCoCo/train2017'
    captions_path = 'data/captions.json'
    split_list_path = 'data/train_split.json'
    vocabulary_path = 'data/vocabulary.json'
    get_train_tfrecord(imgs_dir_path, captions_path, split_list_path, vocabulary_path)


# main()