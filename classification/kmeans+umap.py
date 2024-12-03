import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import argparse
def main(args):
    def get_images_labels(paths):
        labels = [cla for cla in os.listdir(paths)]
        all_labels = []
        all_images = []
        all_names = []
        for label in labels:
            new_parh = os.path.join(paths,label)
            for j in os.listdir(new_parh):
                all_images.append(os.path.join(new_parh,j))
                all_labels.append(int(label))
                all_names.append(j)
        return all_images,all_labels,all_names

    path = args.path
    images_path,labels,all_names = get_images_labels(path)

    features = []
    for path_ , label in zip(images_path,labels):
        img = Image.open(path_)
        img = img.resize((64,64))
        if img.mode !='L':
            img = img.convert('L')
        feature = np.array(img).flatten()
        features.append(feature)

    X = np.array(features)

    #使用UMAP进行降维
    umap_model =umap.UMAP(n_neighbors=15,min_dist=0.1)
    embedding = umap_model.fit_transform(X)
    # print(embedding.shape)

    categories = [0, 100, 1000, 200, 300, 400, 50, 500]
    colors = [ '#F57C6E','#F2B56F','#0A8C78','#88D8DB','#71B7ED','#B8AEED','#4472C4','#956D6D']

    plt.figure(figsize=(12, 10))
    cc = labels
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors[categories.index(l)] for l in cc], cmap='viridis', marker='o', s=5)
    plt.colorbar(scatter, label='Category')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Visualization with Categories')



    plt.text(0.2, 0.95,  '\n'.join([f'{c}: {clr}' for c, clr in zip(categories, colors)]),
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=8)
    plt.show()

    #使用kmeans进行聚类
    kmeans = KMeans(n_clusters=4,random_state=42)
    kmeans.fit(embedding)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    #找出距离每个聚类中心最近的样例
    nearest_indices = []
    for center in centers:
        #计算每个样例到当前聚类中心的距离
        distences = np.linalg.norm(embedding - center , axis=1)
        #找到距离最小的样例的索引
        nearest_index = np.argmin(distences)
        nearest_indices.append(nearest_index)
    dist_paths = []
    for nearest in nearest_indices:
        dist_paths.append(all_names[nearest])
    # import shutil
    # save_path = r'D:\FoodAcademyCooperativeProgram\test\save_parh'
    # for path_one in dist_paths:
    #     label_ = path_one.split('-')[0]
    #     path_res = os.path.join(path,label_,path_one)
    #     save_p = os.path.join(save_path,path_one)
    #     shutil.copy(path_res,save_p)


    colors = ['#F7776E','#6C8FC6','#F6C490','#1DBDC6']
    #可视化UMAP降维后的2D数据
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors[i] for i in labels], cmap='viridis', marker='o',s=5)
    plt.colorbar(label='Cluster Label')
    plt.title('UMAP 2D Embedding with K-means Clustering')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()
    names = []
    for i in all_names:
        names.append(i.split('-')[0])

    import pandas as pd
    # 创建一个DataFrame来存储原始标签和聚类标签
    df = pd.DataFrame({
        'original_label': all_names,
        'cluster_label': labels
    })
    #
    # # 计算每个聚类中每种标签的数量
    result = df.groupby(['cluster_label', 'original_label']).size().unstack(fill_value=0)


    #写入文件中进行记录
    save_path = args.save_path
    result.to_csv(save_path, index=True)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='../data/train')
    parser.add_argument('--save_path',type=str,default='./kmeans+umap/new_KU.csv')


    opt = parser.parse_args()
    main(opt)