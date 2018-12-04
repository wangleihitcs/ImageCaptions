import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import json
import termcolor as tc

import datasets
import metrics
from model import Model

pretrain_inception_v3_ckpt_path = '/home/wanglei/workshop/b_pre_train_model/inception/inception_v3.ckpt'
val_split_path = 'data/val_split.json'      # val size = 5000
test_split_path = 'data/test_split.json'      # val size = 5000
tfrecord_list_path = 'data/tfrecord_list.json'

summary_path = 'data/summary'
model_path_save = 'data/model/my-test'
num_epochs = 1
train_num = 82783

def train():
    md = Model(is_training=True)            # Train model
    md_val = Model(is_training=True)        # Val model
    md_test = Model(is_training=False)      # Test model

    print('---Read Data...')
    image_batch, sentence_batch, mask_batch, image_id_batch = datasets.get_train_batch(tfrecord_list_path, md.batch_size)
    image_list_v, sentence_list_v, mask_list_v, image_id_list_v = datasets.get_inputs(val_split_path)
    image_list_t, sentence_list_t, mask_list_t, image_id_list_t = datasets.get_inputs(test_split_path)

    print('---Training Model...')
    init_fn = slim.assign_from_checkpoint_fn(pretrain_inception_v3_ckpt_path, slim.get_model_variables('InceptionV3'))  # 'InceptionV3'
    saver = tf.train.Saver(max_to_keep=301)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        iter = 1
        loss_list = []
        acc_list = []
        predictions_list = []
        image_id_list = []
        for epoch in range(num_epochs):
            for _ in range(train_num / md.batch_size):
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
                    loss_list, acc_list, predictions_list, image_id_list = eval(sess, md_val, image_list_v, sentence_list_v, mask_list_v, image_id_list_v)
                    bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md_val.batch_size)
                    loss_val = round(np.mean(loss_list), 4)
                    acc_val = round(np.mean(acc_list), 4)
                    print('------epoch = %s, iter = %s, loss = %s, acc = %s, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                                (epoch, iter, tc.colored(loss_val, 'red'), tc.colored(acc_val, 'blue'), tc.colored(bleu, 'red'),
                                 tc.colored(meteor, 'red'), tc.colored(rouge, 'red'), tc.colored(cider, 'red')))

                    # test model
                    loss_list, acc_list, predictions_list, image_id_list = eval(sess, md_test, image_list_t, sentence_list_t, mask_list_t, image_id_list_t)
                    bleu, meteor, rouge, cider = metrics.coco_caption_metrics(predictions_list, image_id_list, batch_size=md_test.batch_size)
                    loss_test = round(np.mean(loss_list), 4)
                    acc_test = round(np.mean(acc_list), 4)
                    print('------epoch = %s, iter = %s, loss = %s, acc = %s, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                          (epoch, iter, tc.colored(loss_test, 'blue'), tc.colored(acc_test, 'blue'), tc.colored(bleu, 'blue'),
                           tc.colored(meteor, 'blue'), tc.colored(rouge, 'blue'), tc.colored(cider, 'blue')))

                    loss_list = []
                    acc_list = []
                    predictions_list = []
                    image_id_list = []

                iter += 1

        coord.request_stop()
        coord.join(threads)

# eval model for val and test
def eval(sess, md, image_list, sentence_list, mask_list, image_id_list):
    loss_list = []
    acc_list = []
    predictions_list = []
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
        loss_list.append(_loss)
        acc_list.append(_acc)
        predictions_list.append(_predictions)
        _image_id_list.append(image_ids)
    return loss_list, acc_list, predictions_list, _image_id_list


train()

