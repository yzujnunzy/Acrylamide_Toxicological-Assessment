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

    #Loading and Preprocessing Images
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

        #Outer layer concentration
        path_concentrations = os.path.join(path,concentration)
        #The inner image is looped
        for path_concentration in os.listdir(path_concentrations):
            path_concentration_imgs = os.path.join(path_concentrations,path_concentration)
            #Smaller images within a single image
            res = Counter()
            images = []
            if len(os.listdir(path_concentration_imgs))==0:
                continue
            for path_concentration_img in os.listdir(path_concentration_imgs):
                r_path_concentration_img = os.path.join(path_concentration_imgs,path_concentration_img)
                images.append(load_and_preprocress_image(r_path_concentration_img))
            # Stacking images as a batch
            images_batch = torch.cat(images, dim=0)
            # Move model and image batches to GPU as well
            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                model = model.cuda()
            #Start forecasting
            with torch.no_grad():
                output = model(images_batch)
            # Get the probability that each image belongs to each category
            _, preds = torch.max(output, 1)

            # Output prediction results
            for pred in preds:
                res[pred]+=1  # Print the predicted category index for each image
            # Use most_common() to find the element with the most occurrences and their counts
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
    parser.add_argument('--weights_path',type=str,default='XXX')
    parser.add_argument('--pre_data',type=str,default='XXX')
    parser.add_argument('--save_path',type=str,default='XXX')

    opt = parser.parse_args()
    main(opt)