# -*- coding: UTF-8 -*-
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import json

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

