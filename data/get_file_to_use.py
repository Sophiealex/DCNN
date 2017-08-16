# import dataUtils
# from gensim import models
import scipy.io as sio
import fileinput
import numpy

# x_, y_, vocabulary, _inv, test_size = dataUtils.load_data()

# model = models.Word2Vec.load('merge48.model.bin')

sent_files = open("merge_all_space.txt", "r", encoding='utf-8')
label_files = open("Binary_Z.txt", "r", encoding='utf-8')
output_data_file = open("data.txt", "w+", encoding='utf-8')
output_test = open("test.txt", "w+", encoding='utf-8')
output_train = open("train.txt", "w+", encoding='utf-8')

all_sent = []
all_label = []
for sent in sent_files:
    all_sent.append(sent)

for label in label_files:
    all_label.append(label)

all_data = [str(x).strip("\n")+":"+y for x,y in zip(all_label,all_sent)]
output_data_file.writelines(all_data)
output_train.writelines(all_data[:-500])
output_test.writelines(all_data[-500:])


sent_files.close()
label_files.close()
output_data_file.close()
output_test.close()
output_train.close()