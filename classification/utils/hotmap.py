import os
import seaborn as sns
import matplotlib.pyplot as plt
#预测结果的顺序是乱的，需要调整顺序为从小到大排列

res = []
with open('../result.txt', 'r', encoding='utf-8') as files:
    for file in files:
        file = file.replace('\n','')
        res.append(file.split(','))
for i in range(9):
    for j in range(9):
        res[i][j] = int(res[i][j])
rea = [0,50,60,100,200,300,400,500,1000]
now = [0,100,1000,200,300,400,50,500,60]
rres = [[] for i in range(9)]
for i in range(9):
    for j in range(9):
        if rea[i]==now[j]:
            for ioo in range(9):
                rres[i].append(res[j][ioo])
            for ii in range(9):
                for ji in range(9):
                    if rea[ii] == now[ji]:
                        rres[i][ii] = res[j][ji]

print(rres)
#创建热力图
plt.figure(figsize= (10,8))
heatmap = sns.heatmap(rres,annot = True,cmap='coolwarm')
# 设置横横坐标轴数字（标签）
labels = ['0', '50', '60', '100', '200', '300', '400', '500', '1000']  # 自定义标签列表
plt.xticks(range(1, len(rres) + 1), labels, rotation=45)  # 设置x轴刻度位置和标签，rotation参数用于旋转标签以避免重叠
plt.yticks(range(1, len(rres) + 1), labels)  # 设置y轴刻度位置和标签

# 添加标题
plt.title('Heatmap of 2D concentrations')
plt.show()
