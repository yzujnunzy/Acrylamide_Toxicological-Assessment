import os
import matplotlib.pyplot as plt

# 设置matplotlib的rcParams，指定字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

path = r"../process/results-6-20240425-105759.txt"
epoch = []
acc = []
loss = []

i = 0
with open(path,'r') as f:
    for words in f:
        if words !='\n':
            if i%3 == 0:
                epoch.append(int(words.strip()[8:-1]))
            elif i%3==1:
                loss.append(float(words.strip()[12:]))
            else:
                acc.append(float(words.strip()[9:]))
            i+=1

plt.plot(epoch,acc,label = "准确率")
plt.plot(epoch,loss,label = "损失函数")
plt.legend() #显示图例
plt.show()
