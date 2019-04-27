# -*- coding: UTF-8 -*-
import tensorflow as tf
import json
import nltk
import os
import numpy as np
from utils import image_utils

def get_train_batch(tfrecord_list_path, batch_size, image_size=224, max_caption_length=20):
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

    # 3. get tf.tfrecord.batch
    image_batch, sentece_batch, mask_batch, image_id_batch = tf.train.shuffle_batch(
        [image, sentence, mask, image_id],
        batch_size=batch_size,
        capacity=15,
        min_after_dequeue = 10
    )

    return image_batch, sentece_batch, mask_batch, image_id_batch

# get data/tfrecord/xxx.tfrecord
def get_train_tfrecord(imgs_path, captions_path, split_list_path, vocabulary_path, image_size=224, max_caption_length=20, mode='train', D=40):
    with open(vocabulary_path, 'r') as file:
        vocabulary = json.load(file)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i
    with open(captions_path, 'r') as file:
        caption_dict = json.load(file)
    with open(split_list_path, 'r') as file:
        split_id_list = json.load(file)

    for i in range(D):
        subsets_num = split_id_list.__len__() / D + 1
        sub_split_id_list = split_id_list[i * subsets_num: (i + 1) * subsets_num]

        tfrecord_name = 'data/tfrecord/' + mode + '-%02d.tfrecord' % i
        writer = tf.python_io.TFRecordWriter(tfrecord_name)

        for image_id in sub_split_id_list:
            filename = str('%012d' % image_id) + '.jpg'
            img_file_path = os.path.join(imgs_path, filename)
            image = image_utils.getImages(img_file_path, image_size)
            image = image.reshape([image_size*image_size*3])

            caption = caption_dict[str(image_id)][0]
            raw_word_list = nltk.word_tokenize(caption)
            word_list = []
            for word in raw_word_list:          # filt out the word not contains in vocabulary
                if vocabulary.__contains__(word):
                    word_list.append(word)
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
        print('%s write to tfrecord success!' % tfrecord_name)
        # break

# get data from memory
def get_inputs(imgs_path, captions_path, split_list_path, vocabulary_path, image_size=224, max_caption_length=20):
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
        raw_word_list = nltk.word_tokenize(caption)
        word_list = []
        for word in raw_word_list:          # filt out the word not contains in vocabulary
            if vocabulary.__contains__(word):
               word_list.append(word)
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

# get 'data/tfrecord_name_train.json' 'data/tfrecord_name_val.json' 'data/tfrecord_name_test.json'
def get_tfrecord_split():
    D = 40
    train_tfrecord_name_list = []
    for i in range(D):
        tfrecord_name = 'data/tfrecord/train-%02d.tfrecord' % i
        train_tfrecord_name_list.append(tfrecord_name)

    D = 2
    val_tfrecord_name_list = []
    for i in range(D):
        tfrecord_name = 'data/tfrecord/val-%02d.tfrecord' % i
        val_tfrecord_name_list.append(tfrecord_name)

    D = 2
    test_tfrecord_name_list = []
    for i in range(D):
        tfrecord_name = 'data/tfrecord/test-%02d.tfrecord' % i
        test_tfrecord_name_list.append(tfrecord_name)

    with open('data/tfrecord_name_train.json', 'w') as file:
        json.dump(train_tfrecord_name_list, file)
    with open('data/tfrecord_name_val.json', 'w') as file:
        json.dump(val_tfrecord_name_list, file)
    with open('data/tfrecord_name_test.json', 'w') as file:
        json.dump(test_tfrecord_name_list, file)
    print('tfrecord split.')


def main():
    imgs_dir_path = '/home/wanglei/workshop/MSCoCo/train2017'
    captions_path = 'data/captions.json'
    train_split_list_path = 'data/image_id_train.json'
    vocabulary_path = 'data/vocabulary.json'
    get_train_tfrecord(imgs_dir_path, captions_path, train_split_list_path, vocabulary_path, mode='train', D=40)

    val_split_list_path = 'data/image_id_val.json'
    # get_train_tfrecord(imgs_dir_path, captions_path, val_split_list_path, vocabulary_path, mode='val', D=2)

    test_split_list_path = 'data/image_id_test.json'
    # get_train_tfrecord(imgs_dir_path, captions_path, test_split_list_path, vocabulary_path, mode='test', D=2)


    # get_tfrecord_split()
# main()
