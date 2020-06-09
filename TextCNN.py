import json
from collections import Counter
import gensim
import numpy as np
import yaml
import jieba
from gensim.models import word2vec
import multiprocessing
import re

# import plaidml.keras
# plaidml.keras.install_backend()
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras
from tensorflow import keras



class Config(object):

    word2VecModel = './word2VecModel_3'
    wordToIndex = './wordJson/wordToIndex.json'
    indexToWord = './wordJson/indexToWord.json'
    modelStructFig = './textCNN_model.png'
    netStructFig = './textCNN_net.png'
    yml = './model/textCNN_8.yml'
    modelCheckpoint = './model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5'
    modelWeights = './model/textCNN_8.h5'
    modelResultsTxt = './textCNN_result_8.txt'

    # 数据集路径
    # stopWordSource = 'empty_stopwords.txt'
    stopWordSource = 'cn_stopwords.txt'
    trainDataSource = 'train.txt'
    validationDataSource = 'validation.txt'
    testDataSource = 'test.txt'

    # 分词后保留大于等于最低词频的词
    miniFreq = 1

    # 统一输入文本序列的定长，取了所有序列长度的均值。超出将被剪断，不足则补0
    sequenceLength = 150
    batchSize = 32
    epochs = 20

    # 生成嵌入词向量的维度
    embeddingSize = 300

    # 卷积核数
    numFilters = 120

    # 卷积核大小
    filterSizes =[1, 2, 3, 4, 5]
    dropoutKeepProb = 0.5

    # L2正则系数
    l2RegLambda = 0.1

# 实例化配置参数对象
config = Config()


# 数据预处理
# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.stopWordSource = config.stopWordSource
        self.trainDataSource = config.trainDataSource
        self.validationDataSource = config.validationDataSource
        self.testDataSource = config.testDataSource

        # 每条输入的系列处理为定长
        self.sequenceLength = config.sequenceLength
        self.embeddingSize = config.embeddingSize

        # self.rate = config.rate
        self.miniFreq = config.miniFreq

        self.stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.validationReviews = []
        self.validationLabels = []

        self.testReviews = []
        self.testLabels = []

        self.wordEmbedding = None
        self.n_symbols = 0

        self.wordToIndex = {}
        self.indexToWord = {}

    def readData(self, filePath):
        text = []
        label = []
        with open(filePath) as file:
            for line in file:
                if line == '\n':
                    continue
                temp = line.replace('\n', '').split('\t')
                temp[1] = ''.join(temp[1].split())
                temp[1] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+|[A-Za-z0-9]+", "", temp[1])
                text.append(temp[1])
                label.append(temp[0])

        # print('data:', len(text), len(label))
        texts = [jieba.lcut(document.replace('\n', '')) for document in text]

        return texts, label

    def read_csv(self, file_name):
        import csv
        text = []
        label = []
        with open(file_name, encoding='utf-8', mode='r') as file:
            # reader = csv.reader(file)
            reader = csv.reader(_.replace('\x00', '') for _ in file)
            for line in reader:
                if line:
                    line[1] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+|[A-Za-z0-9]+", "", line[1])
                    label.append(line[0])
                    text.append(line[1])
        texts = [jieba.lcut(document.replace('\n', '')) for document in text]
        return texts, label

    # 读取停用词
    def readStopWord(self, stopWordPath):
        with open(stopWordPath, 'r') as f:
            stopWords = f.read()
            stopwordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopwordList, list(range(len(stopwordList)))))

    # 按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    def getWordEmbedding(self, words):
        # 中文
        model = gensim.models.Word2Vec.load(config.word2VecModel)

        vocab = []
        wordEmbedding = []

        # 添加"pad"和"unk"
        vocab.append('pad')
        wordEmbedding.append(np.zeros(self.embeddingSize))
        vocab.append('unk')
        wordEmbedding.append(np.random.randn(self.embeddingSize))

        for word in words:
            try:
                vector = model[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + '不存在于词向量中')

        return vocab, np.array(wordEmbedding)

    # 生成词向量和词汇-索引映射词典，可以用全数据集
    def genVocabulary(self, reviews):

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]
        # subWords = allWords

        # 统计词频，排序
        wordCount = Counter(subWords)
        sortWordCount = sorted(wordCount.items(), key = lambda x: x[1], reverse=True)

        # 去掉低频词
        words =[item[0] for item in sortWordCount if item[1] >= self.miniFreq]

        # 获取词列表和顺序对应的预训练权重矩阵
        vocab, wordEmbedding = self.getWordEmbedding(words)

        self.wordEmbedding = wordEmbedding

        self.wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self.indexToWord = dict(zip(list(range(len(vocab))), vocab))
        # self.n_symbols = len(self.wordToIndex) + 1
        self.n_symbols = len(self.wordToIndex) #???

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(config.wordToIndex, 'w', encoding='utf-8') as f:
            json.dump(self.wordToIndex, f)

        with open(config.indexToWord, 'w', encoding='utf-8') as f:
            json.dump(self.indexToWord, f)

    # 将数据集中的每条评论里面的词，根据词表，映射为index表示每条评论 用index组成的定长数组来表示
    def reviewProcess(self, review, sequenceLength, wordToIndex):

        reviewVec = np.zeros(sequenceLength)
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["unk"]

        return reviewVec


    def genTrainEvalData(self, x, y):
        reviews = []
        labels = []

        for i in range(len(x)):
            reviewVec = self.reviewProcess(x[i], self.sequenceLength, self.wordToIndex)
            reviews.append(reviewVec)

            labels.append(y[i])

        Reviews = np.asarray(reviews, dtype='int64')
        Labels = np.array(labels, dtype='float32')
        return Reviews, Labels

    # 初始化训练集和验证集
    def dataGen(self):
        # 读取停用词
        self.readStopWord(self.stopWordSource)

        # 读取数据集
        train_reviews, train_labels = self.readData(self.trainDataSource)
        validation_reviews, validation_labels = self.readData(self.validationDataSource)
        test_reviews, test_labels = self.readData(self.testDataSource)


        # 分词、去停用词
        # 生成 词汇-索引 映射表和预训练权重矩阵，并保存
        # self.genVocabulary(train_reviews + validation_reviews)
        self.genVocabulary(train_reviews)


        # 初始化训练集和测试集
        self.trainReviews,  self.trainLabels = self.genTrainEvalData(train_reviews, train_labels)
        self.validationReviews, self.validationLabels = self.genTrainEvalData(validation_reviews, validation_labels)
        self.testReviews, self.testLabels = self.genTrainEvalData(test_reviews, test_labels)

data = Dataset(config)
data.dataGen()

print("train data shape: {}".format(len(data.trainReviews)))
print("train label shape: {}".format(len(data.trainLabels)))
print('validation data shape: {}'.format(len(data.validationReviews)))
print('validation label shape: {}'.format(len(data.validationLabels)))
print('test data shape shape: {}'.format(len(data.testReviews)))
print('test data label shape: {}'.format(len(data.testLabels)))

# 定义网络结构
def convolution(config):
    sequence_length = config.sequenceLength
    embedding_dimension = config.embeddingSize

    inn = keras.Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    filter_sizes = config.filterSizes
    for size in filter_sizes:
        conv = keras.layers.Conv2D(filters = config.numFilters,
                                   kernel_size = (size, embedding_dimension),
                                   strides = 1,
                                   padding = 'valid',
                                   activation = 'relu')(inn)
        # pool = keras.layers.MaxPool2D(pool_size = (sequence_length - size + 1, 1), padding = 'valid')(conv)
        pool = keras.layers.GlobalMaxPooling2D()(conv)
        cnns.append(pool)
    outt = keras.layers.concatenate(cnns)

    model = keras.Model(inputs = inn, outputs = outt)
    model.summary()
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=config.netStructFig)

    return model

def cnn_mulfilter(n_symbols, embedding_weights, config):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=n_symbols,
                                     output_dim=config.embeddingSize,
                                     weights=[embedding_weights],
                                     input_length=config.sequenceLength))
    model.add(keras.layers.Reshape((config.sequenceLength, config.embeddingSize, 1)))
    model.add(convolution(config))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu', kernel_regularizer=keras.regularizers.l2(config.l2RegLambda)))
    model.add(keras.layers.Dropout(config.dropoutKeepProb))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer = keras.optimizers.Adam(lr=1e-4),
                  # optimizer = tf.keras.optimizers.Adadelta(),
                  loss = keras.losses.binary_crossentropy,
                  metrics = ['accuracy'])
    return model

wordEmbedding = data.wordEmbedding
n_symbols = data.n_symbols
model = cnn_mulfilter(n_symbols, wordEmbedding, config)
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=config.modelStructFig)
model.summary()


# 训练模型
x_train = data.trainReviews
y_train = data.trainLabels
x_validation = data.validationReviews
y_validation = data.validationLabels
x_test = data.testReviews
y_test = data.testLabels


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = keras.callbacks.ModelCheckpoint(config.modelCheckpoint, save_best_only=True, save_weights_only=True)
history = model.fit(x_train,
                    y_train,
                    batch_size=config.batchSize,
                    epochs=config.epochs,
                    validation_data=(x_validation, y_validation),
                    shuffle=True,
                    callbacks=[reduce_lr,early_stopping,model_checkpoint]
                    )
#验证
scores = model.evaluate(x_test, y_test)

# 保存模型
yaml_string = model.to_yaml()
with open(config.yml, 'w') as outfile:
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
# model.save_weights(config.modelWeights)
model.save(config.modelWeights)

print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

with open(config.modelResultsTxt, 'a+') as f:
    f.write('acc\n')
    for item in acc:
        f.write("{}\n".format(item))
    f.write('val_acc\n')
    for item in val_acc:
        f.write("{}\n".format(item))
    f.write('loss\n')
    for item in loss:
        f.write("{}\n".format(item))
    f.write('val_loss\n')
    for item in val_loss:
        f.write("{}\n".format(item))

    f.write('test_loss: %f, accuracy: %f\n' % (scores[0], scores[1]))
