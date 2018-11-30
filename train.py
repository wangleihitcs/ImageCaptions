import tensorflow as tf
from tensorflow.contrib import slim
import termcolor as tc
import numpy as np
import json

import datasets
import metrics
from model import Model

imgs_dir_path = '/home/wanglei/workshop/MSCoCo/train2017'
captions_path = 'data/captions.json'
train_split_path = 'data/train_split.json'  # train size = 82783
val_split_path = 'data/val_split.json'      # val size = 5000
test_split_path = 'data/test_split.json'    # test size = 5000
vocabulary_path = 'data/vocabulary.json'

pretrain_inception_v3_ckpt_path = '/home/wanglei/workshop/b_pre_train_model/inception/inception_v3.ckpt'

start_epoch = 0
end_epoch = 101
summary_path = 'data/summary'
model_path = 'data/model/my-test-300'
model_path_save = 'data/model/my-test'

def get_split_id_list(train_split_path, val_split_path, test_split_path):
    with open(train_split_path, 'r') as file:
        train_split_id_list = json.load(file)
    with open(val_split_path, 'r') as file:
        val_split_id_list = json.load(file)
    with open(test_split_path, 'r') as file:
        test_split_id_list = json.load(file)
    return train_split_id_list, val_split_id_list, test_split_id_list

def train():
    md = Model(is_training=True)    # Train model
    mdv = Model(is_training=True)   # Val model
    mdt = Model(is_training=False)   # Test model

    train_split_id_list, val_split_id_list, test_split_id_list = get_split_id_list(train_split_path, val_split_path, test_split_path)

    print('---Read Dataset...')
    train_num = 82783
    print('train num = %s' % train_num)
    image_list_v, sentence_list_v, mask_list_v, filename_list_v = datasets.get_inputs(imgs_dir_path, captions_path, val_split_id_list, vocabulary_path)
    val_num = image_list_v.__len__()
    print('val num = %s' % val_num)
    image_list_te, sentence_list_te, mask_list_te, filename_list_te = datasets.get_inputs(imgs_dir_path, captions_path, test_split_id_list, vocabulary_path)
    test_num = image_list_v.__len__()
    print('test num = %s' % test_num)


    print('---Training Model...')
    init_fn = slim.assign_from_checkpoint_fn(pretrain_inception_v3_ckpt_path, slim.get_model_variables('InceptionV3'))  # 'InceptionV3'

    saver = tf.train.Saver(max_to_keep=301)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        init_fn(sess)
        # saver.restore(sess, model_path)

        divide_subsets = 3
        for i in range(start_epoch, end_epoch):
            loss_list, acc_list, predictions_list = [], [], []
            sentence_list_metrics = []
            filename_list_tr = []

            for d in range(divide_subsets):
                D = train_num / divide_subsets
                split_id_list = train_split_id_list[d*D:(d+1)*D]
                image_list, sentence_list, mask_list, filename_list = datasets.get_inputs(imgs_dir_path, captions_path, split_id_list, vocabulary_path)

                iters = int(image_list.__len__() / md.batch_size)
                for k in range(iters):
                    images = image_list[k * md.batch_size:(k + 1) * md.batch_size]
                    sentences = sentence_list[k * md.batch_size:(k + 1) * md.batch_size]
                    masks = mask_list[k * md.batch_size:(k + 1) * md.batch_size]
                    filenames = filename_list[k * md.batch_size:(k + 1) * md.batch_size]
                    feed_dict = {md.images: images, md.sentences: sentences,
                                 md.masks: masks}
                    _, _summary, _global_step, _loss, _acc, _predictions, = sess.run(
                        [md.step_op, md.summary, md.global_step, md.loss, md.accuracy, md.predictions], feed_dict=feed_dict)
                    train_writer.add_summary(_summary, _global_step)

                    loss_list.append(_loss)
                    acc_list.append(_acc)
                    predictions_list.append(_predictions)
                    sentence_list_metrics += sentences
                    filename_list_tr += filenames

            if i % 1 == 0:
                bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, sentence_list_metrics, filename_list_tr, vocabulary_path, batch_size=md.batch_size)
                print('epoch = %s, loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                      (i, np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

                # val model
                loss_list, acc_list, predictions_list = [], [], []
                sentence_list_metrics = []
                iters = int(val_num / md.batch_size)
                for k in range(iters):
                    images = image_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    sentences = sentence_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    masks = mask_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    feed_dict = {mdv.images: images, mdv.sentences: sentences,
                                 mdv.masks: masks}
                    _loss, _acc, _predictions, = sess.run(
                        [mdv.loss, mdv.accuracy, mdv.predictions],
                        feed_dict=feed_dict)

                    loss_list.append(_loss)
                    acc_list.append(_acc)
                    predictions_list.append(_predictions)
                    sentence_list_metrics += sentences

                bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, sentence_list_metrics, filename_list_v, vocabulary_path, batch_size=mdv.batch_size)
                loss_val = round(np.mean(loss_list), 4)
                print('------epoch = %s, loss = %s, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                      (i, tc.colored(loss_val, 'red'), np.mean(acc_list), tc.colored(bleu, 'red'), tc.colored(meteor, 'red'), tc.colored(rouge, 'red'),
                       tc.colored(cider, 'red')))

                # test model
                loss_list, acc_list, predictions_list = [], [], []
                sentence_list_metrics = []
                iters = int(val_num / md.batch_size)
                for k in range(iters):
                    images = image_list_te[k * md.batch_size:(k + 1) * md.batch_size]
                    sentences = sentence_list_te[k * md.batch_size:(k + 1) * md.batch_size]
                    masks = mask_list_te[k * md.batch_size:(k + 1) * md.batch_size]
                    feed_dict = {mdt.images: images, mdt.sentences: sentences,
                                 mdt.masks: masks}
                    _loss, _acc, _predictions, = sess.run(
                        [mdt.loss, mdt.accuracy, mdt.predictions],
                        feed_dict=feed_dict)

                    loss_list.append(_loss)
                    acc_list.append(_acc)
                    predictions_list.append(_predictions)
                    sentence_list_metrics += sentences

                bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, sentence_list_metrics, filename_list_te, vocabulary_path, batch_size=mdt.batch_size, is_training=mdt.is_training)
                loss_test = round(np.mean(loss_list), 4)
                print('------epoch = %s, loss = %s, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                      (i, loss_test, np.mean(acc_list), tc.colored(bleu, 'blue'), tc.colored(meteor, 'blue'), tc.colored(rouge,'blue'),
                       tc.colored(cider, 'blue')))

            if i % 1 == 0:
                saver.save(sess, model_path_save, global_step=i)

        train_writer.close()
        print('---Training complete.')
train()
