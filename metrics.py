# -*- coding: UTF-8 -*-
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import json

def coco_caption_metrics(predictions_list, image_id_list, vocabulary_path='data/vocabulary.json', max_caption_length=25, batch_size=32, is_training=True):
    with open(vocabulary_path, 'r') as file:
        vocabulary_list = json.load(file)
    word2id = {}
    for i in range(vocabulary_list.__len__()):
        word2id[vocabulary_list[i]] = i
    id2word = {v: k for k, v in word2id.items()}

    with open('data/captions_gt.json', 'r') as file:
        captions_gt_dict = json.load(file)

    gts = {}
    res = {}
    for i in range(0, predictions_list.__len__()):
        for j in range(0, batch_size):
            sen_input, sen_ground_truth = [], []
            for k in range(max_caption_length):
                id_input = int(predictions_list[i][k][j])
                sen_input.append(id2word[id_input])

            sen_pre = []
            for n in range(max_caption_length):
                word = sen_input[n]
                if word != '</S>':
                    sen_pre.append(word)
                else:
                    break

            str_input = ' '.join(sen_pre)
            image_id = image_id_list[i][j][0]

            # print(image_id)
            res[image_id] = [str_input]
            gts[image_id] = captions_gt_dict[str(image_id)]

    if not is_training:
        # for key in gts.keys():
        #     str_input = res[key]
        #     str_grundtruth = gts[key]
        #     print(key)
        #     print(str_input)
        #     print(str_grundtruth)
        #     print('*' * 100)

        with open('data/result/result_res.json', 'w') as file:
            json.dump(res, file)
        with open('data/result/result_gts.json', 'w') as file:
            json.dump(gts, file)
        # print('result.json get success')

    bleu_scorer = Bleu(n=4)
    bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)

    rouge_scorer = Rouge()
    rouge, _ = rouge_scorer.compute_score(gts=gts, res=res)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts=gts, res=res)

    meteor_scorer = Meteor()
    meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)

    for i in range(4):
        bleu[i] = round(bleu[i], 4)
    return bleu, round(meteor, 4), round(rouge, 4), round(cider, 4)


# eval predictions and ground truth by coco captions metrics
def eval(result_gts_path, result_res_path):
    with open(result_gts_path, 'r') as file:
        gts_dict = json.load(file)
    with open(result_res_path, 'r') as file:
        res_dict = json.load(file)

    bleu_score = Bleu(n=4)
    bleu, _ = bleu_score.compute_score(gts=gts_dict, res=res_dict)

    meteor_score = Meteor()
    meteor, _ = meteor_score.compute_score(gts=gts_dict, res=res_dict)

    rouge_scorer = Rouge()
    rouge, _ = rouge_scorer.compute_score(gts=gts_dict, res=res_dict)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts=gts_dict, res=res_dict)

    return bleu, meteor, rouge, cider

