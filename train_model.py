import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM, Dropout, Bidirectional
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix


# Extract lable and words list from file
input_sentence = [i.split("::")[1].replace("\n", '') for i in open("pre_train_input.txt", encoding="utf8")]
input_ans = [i.split("::")[1].replace("\n", '') for i in open("pre_train_ans.txt", encoding="utf-8")]

input_zip = list(zip(input_sentence, input_ans))
shuffle(input_zip)

# find all available vocab
words = [[w for w in deepcut.tokenize(s[0]) if w != ' '] for s in input_zip]

# Extracted only the word around "เขา" not longer than 21 word
def split_from_kau(words):
    q = []
    flag_found_kau = False
    for w in words:
        q.append(w)
        if "เขา" in w:
            flag_found_kau = True
        elif flag_found_kau:
            if len(q) >= 21:
                break
        else:
            if len(q) > 10:
                q.pop(0)
    return q

# print(index_of_kau(words[0]))
words_split = [split_from_kau(w) for w in words]
# print(words_split[-10:])
max_sentence_length = max([len(s) for s in words_split]) # should be 21
# print(max_sentence_length)


vocab = set([w for s in words_split for w in s])
# use pretrain word vector cc.th.300.vec
pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
count = 0
vocab_vec = {}
for line in pretrained_word_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab): 
            vocab_vec[line[0]] = line[1:]
    count = count + 1


# ไม่เจอคำที่มีให้ข้าม
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

# print(word_vectors.shape)
# print(word_vectors[:2])

# CREATE MODEL !!!
inputLayer = Input(shape=(max_sentence_length,word_vector_length,))
# rnn = GRU(30, activation='relu')(inputLayer)
rnn = Bidirectional(LSTM(30, activation='relu'))(inputLayer)
rnn = Dropout(0.5)(rnn)
outputLayer = Dense(3, activation='softmax')(rnn) # for 3 classes
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())


# convert from H P M to 0 1 2
dict_label = {"H":0,"P":1,"M":2}
labels = [dict_label[l[1]] for l in input_zip]
# print(labels)

# train model
history = model.fit(word_vectors, to_categorical(labels), epochs=200, batch_size=50, validation_split = 0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()


# check accuracy from every sentence
y_pred = model.predict(word_vectors)

cm = confusion_matrix(labels, y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)
print((cm[0,0]+cm[1,1]+cm[2,2]) / sum([cm[i,j] for i in range(3) for j in range(3)]))

model.save("model.h5")