# 训练数据预处理
import numpy as np
import os
import matplotlib.pylab as plt
import re

# 评论语料目录
trainDataSource = 'train.txt'
validationDataSource = 'validation.txt'
testDataSource = 'test.txt'

def read(dataSourcePath):
    sentences = []
    with open(dataSourcePath) as file:
        for line in file:
            if line == '\n':
                continue
            temp = line.replace('\n', '').split('\t')
            temp[1] = ''.join(temp[1].split())
            temp[1] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+|[A-Za-z0-9]+", "", temp[1])
            sentences.append(temp[1])
    return sentences

sentences = read(trainDataSource) + read(validationDataSource) + read(testDataSource)

# 获取所有文本的长度
all_length = [len(i) for i in sentences]

# 可视化预料序列长度，可见有99.61%的文本长度都在150以下
plt.hist(all_length, bins=30)
plt.ylabel('Number of sentences')
plt.xlabel('Text length')
plt.title('Text length distribution in Corpus')
plt.show()

length1 = np.mean(np.array(all_length) <= 50)
length2 = np.mean(np.array(all_length) <= 100) - length1
length3 = np.mean(np.array(all_length) <= 150) - length2
length4 = np.mean(np.array(all_length) > 150)
# print(np.mean(np.array(all_length) <= 50))
# print(np.mean(np.array(all_length) <= 100))
# print(np.mean(np.array(all_length) <= 150))
# print(np.mean(np.array(all_length) > 150))
print(length1, length2, length3, length4)

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('1.pdf')

labels = ['0 <= length <= 50','50 < length <=100','100 < length <= 150','length > 150']
sizes = [0.022926604092937376,0.7886213263579013,0.20753192798892128,0.003846745653177412]
explode = (0,0,0,0.2)
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.2f%%',shadow=False,startangle=150)
plt.title('Text length distribution in Corpus')
# plt.show()
plt.tight_layout()
pdf.savefig()
plt.close()
pdf.close()
