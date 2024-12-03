'''
1)Import the relevant packages and load the model
'''

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from models.resnet import resnet34
from models.densenet import densenet121
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def cam_show_img(img, feature_map, name,ma):

    cam_img = 0.5 * feature_map + 0.5 * img
    if not os.path.exists(name):
        os.makedirs(name)
    path_cam_img = os.path.join(name, ma)
    cv2.imwrite(path_cam_img, cam_img)

def main(args):
    paths = args.inpath
    json_path = 'class_indices.json'
    path_imgs = os.listdir(paths)
    # Loading squeezenet1_1 pre-trained model
    net = resnet34(num_class=8, attention=True)
    model_weight_path = 'XXX'
    # net = densenet121(num_classes = 8)
    # model_weight_path = './weights/densenet_121_1.pth'
    net.load_state_dict(torch.load(model_weight_path), strict=False)
    net.eval()  # 8
    #target_layer = [net.layer2]
    target_layer = [net.features.denseblock3.denselayer18.conv2]
    for image_path in path_imgs:
        path_img = os.path.join(paths, image_path)
        rgb_img = cv2.imread(path_img, 1)  # imread() reads in BGR format
        rgb_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize((224,224)),

        ])
        rgb_img1 = transform(rgb_img)
        rgb_img = np.float32(rgb_img1) / 255
        img = cv2.imread(path_img, 1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])  # torch.Size([1, 3, 224, 224])


        cam = GradCAM(model=net, target_layers=target_layer)
        grayscale_cam = cam(input_tensor=input_tensor)  # [batch, 224,224]
        grayscale_cam = grayscale_cam[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)

        path = args.outpath
        os.makedirs(path, exist_ok=True)
        pp = os.path.join(path,image_path)
        cv2.imwrite(pp,visualization)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath', type=str,
                        default=r"XXX")
    parser.add_argument('--outpath',type=str,default='XXX')


    opt = parser.parse_args()

    main(opt)
