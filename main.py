import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import json
import termcolor as tc

import datasets
import metrics
from cnn_rnn_model import Model

imgs_path = 'data/train2017'
captions_path = 'data/captions.json'
val_split_path = 'data/image_id_val.json'              # val size = 5000
test_split_path = 'data/image_id_test.json'            # test size = 5000

train_tfrecord_name_path = 'data/tfrecord_name_train.json'
val_tfrecord_name_path = 'data/tfrecord_name_val.json'
test_tfrecord_name_path = 'data/tfrecord_name_test.json'
vocabulary_path = 'data/vocabulary.json'

pretrain_inception_v3_ckpt_path = 'data/inception/inception_v3.ckpt'

summary_path = 'data/summary'                       # data/summary to save events tf.summary
model_path_save = 'data/model/my-test'              # data/model to save my-test-xxx.ckpt
num_epochs = 101                                    # tfrecord how many epochs
train_num = 82783                                   # tfrecord size = 82783
val_num = 5000
test_num = 5000

def train():
    md = Model(is_training=True)            # Train model
    # md_val = Model(is_training=True)        # Val model
    # md_test = Model(is_training=False)      # Test model

    print('---Read Data...')
    image_batch, sentence_batch, mask_batch, image_id_batch = datasets.get_train_batch(train_tfrecord_name_path, md.batch_size)
    # image_list_v, sentence_list_v, mask_list_v, image_id_list_v = datasets.get_inputs(imgs_path, captions_path, val_split_path, vocabulary_path)
    # image_list_t, sentence_list_t, mask_list_t, image_id_list_t = datasets.get_inputs(imgs_path, captions_path, test_split_path, vocabulary_path)

    print('---Training Model...')
    init_fn = slim.assign_from_checkpoint_fn(pretrain_inception_v3_ckpt_path, slim.get_model_variables('InceptionV3'))  # 'InceptionV3'
    saver = tf.train.Saver(max_to_keep=301)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        init_fn(sess)

        coord = tf.train.Coordinator()                          # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        iter = 1
        loss_list = []
        acc_list = []
        predictions_list = []
        image_id_list = []
        for epoch in range(num_epochs):
            for _ in range(train_num / md.batch_size):
                # for efficiency, it is no need tensor to numpy, the numpy to tensor, so no need feed_dict, but for model's simplicity, I make the unneccessary move.
                images, sentences, masks, image_ids = sess.run([image_batch, sentence_batch, mask_batch, image_id_batch])

                feed_dict = {md.images: images,
                             md.sentences: sentences,
                             md.masks: masks}

                _, _summary, _global_step, _loss, _acc, _predictions, = sess.run(
                    [md.step_op, md.summary, md.global_step, md.loss, md.accuracy, md.predictions], feed_dict=feed_dict)
                train_writer.add_summary(_summary, _global_step)

                loss_list.append(_loss)
                acc_list.append(_acc)
                predictions_list.append(_predictions)
                image_id_list.append(image_ids)
                if iter % 1000 == 0:
                    saver.save(sess, model_path_save, global_step=iter)

                    bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md.batch_size)
                    print('epoch = %s, iter = %s, loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                          (epoch, iter, np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

                    # val model
                    # loss_list, acc_list, predictions_list, image_id_list = eval(sess, md_val, image_list_v, sentence_list_v, mask_list_v, image_id_list_v)
                    # bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md_val.batch_size)
                    # loss_val = round(np.mean(loss_list), 4)
                    # acc_val = round(np.mean(acc_list), 4)
                    # print('------epoch = %s, iter = %s, loss = %s, acc = %s, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                    #             (epoch, iter, tc.colored(loss_val, 'red'), tc.colored(acc_val, 'red'), tc.colored(bleu, 'red'),
                    #              tc.colored(meteor, 'red'), tc.colored(rouge, 'red'), tc.colored(cider, 'red')))

                    # # test model
                    # loss_list, acc_list, predictions_list, image_id_list = eval(sess, md_test, image_list_t, sentence_list_t, mask_list_t, image_id_list_t)
                    # bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md_test.batch_size)
                    # loss_test = round(np.mean(loss_list), 4)
                    # acc_test = round(np.mean(acc_list), 4)
                    # print('------epoch = %s, iter = %s, loss = %s, acc = %s, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                    #       (epoch, iter, tc.colored(loss_test, 'blue'), tc.colored(acc_test, 'blue'), tc.colored(bleu, 'blue'),
                    #        tc.colored(meteor, 'blue'), tc.colored(rouge, 'blue'), tc.colored(cider, 'blue')))

                    loss_list = []
                    acc_list = []
                    predictions_list = []
                    image_id_list = []

                iter += 1

        coord.request_stop()
        coord.join(threads)

# eval model for val and test
def eval(sess, md, image_list, sentence_list, mask_list, image_id_list):
    _loss_list = []
    _acc_list = []
    _predictions_list = []
    _image_id_list = []

    batch_size = md.batch_size
    iters = int(image_id_list.__len__() / batch_size)
    for k in range(iters):
        images = image_list[k * batch_size:(k + 1) * batch_size]
        sentences = sentence_list[k * batch_size:(k + 1) * batch_size]
        masks = mask_list[k * batch_size:(k + 1) * batch_size]
        image_ids = image_id_list[k * batch_size:(k + 1) * batch_size]
        feed_dict = {md.images: images,
                     md.sentences: sentences,
                     md.masks: masks}
        _loss, _acc, _predictions, = sess.run([md.loss, md.accuracy, md.predictions], feed_dict=feed_dict)
        _loss_list.append(_loss)
        _acc_list.append(_acc)
        _predictions_list.append(_predictions)
        _image_id_list.append(image_ids)

    return _loss_list, _acc_list, _predictions_list, _image_id_list


def test(tfrecord_list_path, data_nums, model_path):
    md = Model(is_training=False)  # Test model

    print('---Read Data...')
    image_batch, sentence_batch, mask_batch, image_id_batch = datasets.get_train_batch(tfrecord_list_path, md.batch_size)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, model_path)

        coord = tf.train.Coordinator()  # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        loss_list = []
        acc_list = []
        predictions_list = []
        image_id_list = []
        for _ in range(data_nums / md.batch_size):
            images, sentences, masks, image_ids = sess.run([image_batch, sentence_batch, mask_batch, image_id_batch])

            feed_dict = {md.images: images,
                         md.sentences: sentences,
                         md.masks: masks}

            _loss, _acc, _predictions, = sess.run([md.loss, md.accuracy, md.predictions], feed_dict=feed_dict)
            loss_list.append(_loss)
            acc_list.append(_acc)
            predictions_list.append(_predictions)
            image_id_list.append(image_ids)

        bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md.batch_size, is_training=md.is_training)
        print('loss = %s, acc = %s, belu = %s, meteor = %s, rouge = %s, cider = %s' %
              (np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

        coord.request_stop()
        coord.join(threads)

train()

model_path = 'data/model/my-test-10000'
# test(train_tfrecord_name_path, train_num, model_path)
# test(val_tfrecord_name_path, val_num, model_path)
# test(test_tfrecord_name_path, test_num, model_path)

