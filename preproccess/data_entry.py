import json
import random
import string

captions_path = '../data/coco_annotations/captions_train2017.json'

with open(captions_path, 'r') as file:
    captions = json.load(file)

gts_dict = {}
captions_dict = {}
image_id_list = []
for image_dict in captions['images']:
    image_id = image_dict['id']
    image_id_list.append(image_id)
    gts_dict[image_id] = []
    captions_dict[image_id] = []

_filter = string.punctuation + '0123456789'
for annotation in captions['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']
    caption = caption.lower()
    caption = caption.replace('\n', '')
    post_caption = ''.join(c for c in caption if c not in _filter)

    gts_dict[image_id].append(post_caption)

for image_id in gts_dict.keys():
    k = random.randint(0, 4)
    captions_dict[image_id] = [gts_dict[image_id][k]]

with open('../data/captions_gt.json', 'w') as file:
    json.dump(gts_dict, file)
with open('../data/captions.json', 'w') as file:
    json.dump(captions_dict, file)
print('dataset captions get success!')

