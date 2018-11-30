import json
import nltk
import collections
import random

vocabulary = ['<S>', '</S>', '<EOS>']
captions_path = '../data/captions.json'
vocabulary_path = '../data/vocabulary.json'

with open(captions_path, 'r') as file:
    captions_dict = json.load(file)

raw_word_list = []
for key in captions_dict.keys():
    caption = captions_dict[key][0]
    words = nltk.word_tokenize(caption)
    raw_word_list += words
print('raw word size = %s' % raw_word_list.__len__())

raw_word_count = collections.Counter(raw_word_list)
print('raw word count size = %s' % raw_word_count.__len__())

word_list = []
for key in raw_word_count.keys():
    word, num = key, raw_word_count[key]
    if num >= 1 and not vocabulary.__contains__(word):
        word_list.append(word)
print('word occur times >= 2 size = %s' % word_list.__len__())
random.shuffle(word_list)
vocabulary += word_list

# word_list = raw_word_count.most_common(9989)
with open(vocabulary_path, 'w') as file:
    json.dump(vocabulary, file)
print('vocabulary.json get success.')



