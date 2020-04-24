from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import deepcut
import codecs

# Read Input and sequence number
read_input = [i.replace("\n", '') for i in open("input.txt", encoding="utf8")]

input_sentence = [r.split("::")[1] for r in  read_input]
input_sentence_num = [r.split("::")[0] for r in  read_input]

# tokenize
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in input_sentence]

# Extracted only the word around "เขา" not longer than 21 word
def split_from_kau(words):
    q = []
    flag_found_kau = False
    after_found = 0
    for w in words:
        q.append(w)
        if "เขา" in w:
            flag_found_kau = True
            after_found = len(q)
        elif flag_found_kau:
            if len(q) >= 21:
                break
        else:
            if len(q) > 10:
                q.pop(0)
    return q

words_split = [split_from_kau(w) for w in words]
max_sentence_length = 21

# use pretrain word vector cc.th.300.vec
vocab = set([w for s in words_split for w in s])
pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
count = 0
vocab_vec = {}
for line in pretrained_word_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab): 
            vocab_vec[line[0]] = line[1:]
    count = count + 1

# fill the initial with 0
word_vector_length = 300
word_vectors = np.zeros((len(words_split),max_sentence_length,word_vector_length))
sample_count = 0
for s in words_split:
    word_count = 0
    for w in s:
        try:
            word_vectors[sample_count,max_sentence_length-word_count-1,:] = vocab_vec[w]
            word_count = word_count+1
        except:
            pass
    sample_count = sample_count+1


# create dict of H P M and its reverse
dict_label = {"H":0,"P":1,"M":2}
dict_label_reverse = {v:k for k,v in dict_label.items()}
# print(dict_label)
# print(dict_label_reverse)


# Load model !!!
model = load_model('model.h5')

model.summary()

# predict from input !!!
y_pred = model.predict(word_vectors)

# convert 0 1 2 back to H P M
def find_index_of_max(l):
    i = (0,l[0])
    for y in enumerate(l):
        if i[1] < y[1]:
            i = y
    return i[0]

labels = [dict_label_reverse[find_index_of_max(pred)] for pred in y_pred]
# print(labels[:10])

# write result
with codecs.open("ans.txt", "w", "utf-8") as writer:
    for res in zip(input_sentence_num, labels):
        writer.write(res[0]+"::"+res[1]+"\n")
    