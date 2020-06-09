# 预训练词向量
import re
import jieba
from gensim.models import word2vec
import multiprocessing


# 评论语料目录
trainDataSource = 'train.txt'
validationDataSource = 'validation.txt'

def read(dataSourcePath):
    sentences = []
    with open(dataSourcePath) as file:
        for line in file:
            if line == '\n':
                continue
            temp = line.replace('\n', '').split('\t')
            temp[1] = ''.join(temp[1].split())
            temp[1] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+|[A-Za-z0-9]+", "", temp[1])
            # sentences.append(temp[1])
            sentences.append(jieba.lcut(temp[1]))
    return sentences

sentences_train = read(trainDataSource)
sentences_train_validation = read(trainDataSource) + read(validationDataSource)

embeddingSize = 300
miniFreq = 1

word2VecModel_1 = word2vec.Word2Vec(sentences = sentences_train, size = embeddingSize,
    min_count = miniFreq, window = 10, workers = multiprocessing.cpu_count(), sg = 0, iter = 20)
word2VecModel_1.save('word2VecModel_1')

word2VecModel_2 = word2vec.Word2Vec(sentences = sentences_train_validation, size = embeddingSize,
    min_count = miniFreq, window = 10, workers = multiprocessing.cpu_count(), sg = 0, iter = 20)
word2VecModel_2.save('word2VecModel_2')

word2VecModel_3 = word2vec.Word2Vec(sentences = sentences_train, size = embeddingSize,
    min_count = miniFreq, window = 10, workers = multiprocessing.cpu_count(), sg = 1, iter = 20)
word2VecModel_3.save('word2VecModel_3')

word2VecModel_4 = word2vec.Word2Vec(sentences = sentences_train_validation, size = embeddingSize,
    min_count = miniFreq, window = 10, workers = multiprocessing.cpu_count(), sg = 1, iter = 20)
word2VecModel_4.save('word2VecModel_4')

import gensim
word2VecModel = 'word2VecModel_1'
model = gensim.models.Word2Vec.load(word2VecModel)

# print(model.wv.vocab.keys())
print(model['烂俗'])
