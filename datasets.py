# -*- coding: UTF-8 -*-
import json
import nltk
import os
import numpy as np
from utils import image_utils

# get train data、val data、test data
def get_inputs(imgs_path, captions_path, split_id_list, vocabulary_path, image_size=224, max_caption_length=25):
    vocabulary = get_vocabulary(vocabulary_path)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i

    with open(captions_path, 'r') as file:
        caption_dict = json.load(file)

    image_list, sentence_list, mask_list, filename_list = [], [], [], []
    for image_id in split_id_list:
        filename = str('%012d' % image_id) + '.jpg'
        img_file_path = os.path.join(imgs_path, filename)
        image_list.append(image_utils.getImages(img_file_path, image_size))

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

        filename_list.append(str(image_id))

    return image_list, sentence_list, mask_list, filename_list

def get_inputs_batch(t, batch_size, imgs_path, captions_path, split_path, vocabulary_path, image_size=224, max_caption_length=25):
    vocabulary = get_vocabulary(vocabulary_path)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i

    with open(captions_path, 'r') as file:
        caption_dict = json.load(file)
    with open(split_path, 'r') as file:
        split_id_list = json.load(file)

    image_list, sentence_list, mask_list, filename_list = [], [], [], []
    for image_id in split_id_list[t*batch_size:(t+1)*batch_size]:
        filename = str('%012d' % image_id) + '.jpg'
        img_file_path = os.path.join(imgs_path, filename)
        image_list.append(image_utils.getImages(img_file_path, image_size))

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

        filename_list.append(str(image_id))

    return image_list, sentence_list, mask_list, filename_list

# get vocabulary
def get_vocabulary(vocabulary_path):
    with open(vocabulary_path, 'r') as file:
        vocabulary = json.load(file)
    return vocabulary

def main():
    captions_path = 'data/annotations/captions_val2017.json'


# main()