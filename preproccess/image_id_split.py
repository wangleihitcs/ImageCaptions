import json
import random

captions_path = '../data/captions_train2017.json'

with open(captions_path, 'r') as file:
    train_dict = json.load(file)

# all image size = 118287
image_id_list = []
for image_dict in train_dict['images']:
    image_id = image_dict['id']
    image_id_list.append(image_id)

train_id_list = random.sample(image_id_list, 82783)

remain_list1 = list(set(image_id_list) - set(train_id_list))
val_id_list = random.sample(remain_list1, 5000)

remain_list2 = list(set(remain_list1) - set(val_id_list))
test_id_list = random.sample(remain_list2, 5000)

print(image_id_list.__len__())
print(train_id_list.__len__())
print(val_id_list.__len__())
print(test_id_list.__len__())

with open('../data/image_id_train.json', 'w') as file:
    json.dump(train_id_list, file)
with open('../data/image_id_val.json', 'w') as file:
    json.dump(val_id_list, file)
with open('../data/image_id_test.json', 'w') as file:
    json.dump(test_id_list, file)
print('split success!')
