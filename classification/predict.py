import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from collections import Counter
import utils
from models.resnet import resnet34
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    model = resnet34(num_class=8).to(device)

    # load model weights
    weights_path = args.weights_path
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # prediction
    model.eval()
    # load image
    path = args.pre_data
    concentrations = os.listdir(path)
    result = [[0]*9 for i in range(9)]

    #加载与预处理图片
    def load_and_preprocress_image(image_path):
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = Image.open(image_path)
        img = data_transform(img)
        img = torch.unsqueeze(img,dim = 0)
        return img

    for i,concentration in enumerate(concentrations):

        #外层浓度
        path_concentrations = os.path.join(path,concentration)
        #里层图片进行循环
        for path_concentration in os.listdir(path_concentrations):
            path_concentration_imgs = os.path.join(path_concentrations,path_concentration)
            #单个图片中的小图
            res = Counter()
            images = []
            if len(os.listdir(path_concentration_imgs))==0:
                continue
            for path_concentration_img in os.listdir(path_concentration_imgs):
                r_path_concentration_img = os.path.join(path_concentration_imgs,path_concentration_img)
                images.append(load_and_preprocress_image(r_path_concentration_img))
            # 堆叠图片为一个批次
            images_batch = torch.cat(images, dim=0)
            # 把模型和图片批次也移动到GPU
            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                model = model.cuda()
            #开始预测
            with torch.no_grad():
                output = model(images_batch)
            # 获取每个图片属于各个类别的概率
            _, preds = torch.max(output, 1)

            # 输出预测结果
            for pred in preds:
                res[pred]+=1  # 打印每个图片的预测类别索引
            # 使用most_common()找到出现次数最多的元素及其次数
            most_common_element = res.most_common(1)[0][0]
            result[i][int(most_common_element)]+=1
    rea = [0, 50, 100, 200, 300, 400, 500, 1000]
    now = [0, 100, 1000, 200, 300, 400, 50, 500]
    rres = [[] for i in range(9)]
    for i in range(9):
        for j in range(9):
            if rea[i] == now[j]:
                for ioo in range(9):
                    rres[i].append(result[j][ioo])
                for ii in range(9):
                    for ji in range(9):
                        if rea[ii] == now[ji]:
                            rres[i][ii] = result[j][ji]


    with open(args.save_path,'a') as f:
        for i in result:
            for j in i:
                f.write(str(j)+' ')
            f.write("\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path',type=str,default='./weights/resnet_34.pth')
    parser.add_argument('--pre_data',type=str,default='../data/val')
    parser.add_argument('--save_path',type=str,default='./result.txt')

    opt = parser.parse_args()
    main(opt)