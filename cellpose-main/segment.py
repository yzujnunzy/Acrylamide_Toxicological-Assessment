import cv2
import os
import numpy as np
import os
import skimage.io as io
from cellpose_ import models , core
from PIL import Image
use_GPU = core.use_gpu()
from cellpose.io import logger_setup
logger_setup()
import argparse

def main(args):
    # 设置图像文件夹路径和保存结果的文件夹路径
    #原图
    path = args.path
    #标签文件路径
    masks_path=args.masks_path
    #原图文件路径
    image_folder = args.image_folder
    # 存放分割结果的文件夹路径（每种浓度结果存放一起）
    output_folder = args.output_folder
    # 存放单个细胞分割文件路径（每张图片结果单独存放）
    output_fen = args.output_fen


    #原图->标签
    os.makedirs(masks_path, exist_ok=True)
    filenames =os.listdir(path)
    #加载模型
    model = models.Cellpose(gpu=use_GPU, model_type='cyto',nchan=3)
    channels = [1,1]
    for filename in filenames:
      print(f"begin:{filename}")
      files = []
      paths = os.path.join(path,filename)
      #创建目录
      masks_t=os.path.join(masks_path,filename)
      os.makedirs(masks_t, exist_ok=True)
      path_t = os.listdir(paths)
      for img_path in path_t:
         files.append(os.path.join(paths,img_path))
      imgs = [io.imread(f) for f in files]
      masks, flows, styles, diams = model.eval(imgs, diameter=None, flow_threshold=None)
      for i in range(len(masks)):
        data_2d = masks[i]
        data_3d = np.dstack((data_2d * 255, np.zeros_like(data_2d), np.zeros_like(data_2d)))

        data_3d = data_3d.astype(np.uint8)
        mask = Image.fromarray(data_3d, 'RGB')
        save_files=os.path.join(masks_t,path_t[i])
        mask.save(save_files)

    #标签分割
    for image_files in os.listdir(masks_path):
        output_folder_w=os.path.join(output_folder,image_files)
        os.makedirs(output_folder_w,exist_ok=True)
        mask_files_w = os.path.join(image_folder,image_files)
        image_files_w=os.path.join(masks_path,image_files)
        f_image_files_w=os.path.join(output_fen,image_files)
        os.makedirs(f_image_files_w, exist_ok=True)
        for image_file in os.listdir(image_files_w):
            #读取图像
            ss = image_file.split('-')
            # img_file = ss[0]+'-'+ss[1]+'-w2.tif'
            img_file =image_file
            mask_path = os.path.join(mask_files_w, img_file)
            img_path = os.path.join(image_files_w, image_file)
            fen_img_path = os.path.join(f_image_files_w, img_file[:-4])
            os.makedirs(fen_img_path, exist_ok=True)
            img = cv2.imread(img_path)
            image = cv2.imread(mask_path)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

            # 计算梯度的绝对值，并将其转换回 8 位
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            # 合并梯度（近似）
            edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            # 获取每个细胞的轮廓信息
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 保存每个细胞的位置信息
            cell_positions = []

            for i, contour in enumerate(contours):
                # 获取每个细胞的边界框
                x, y, w, h = cv2.boundingRect(contour)
                if w>=10 and h>=10 and w<45 and h<45:
                    # 根据坐标信息进行图像切割
                    cropped_image = image[y:y + h, x:x + w]
                    output_image_path = os.path.join(output_folder_w, f'{img_file[:-4]}_{x}_{y}.jpg')
                    output_image_path_ = os.path.join(fen_img_path, f'{img_file[:-4]}_{x}_{y}.jpg')
                    cv2.imwrite(output_image_path, cropped_image)
                    cv2.imwrite(output_image_path_, cropped_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,default=r"../data/picture")
    parser.add_argument('--masks_path', type=str,default=r"../data/mask")
    parser.add_argument('--image_folder', type=str,default=r"../data/new_picture")
    parser.add_argument('--output_folder',type=str,default='../data/seg')
    parser.add_argument('--output_fen',type=str,default='../data/seg_all')


    opt = parser.parse_args()

    main(opt)

