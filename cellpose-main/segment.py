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
    #Setting the image folder path and the folder path for saving results
    #Raw picture
    path = args.path
    #Label File Path
    masks_path=args.masks_path
    #Original File Path
    image_folder = args.image_folder
    # Path to the folder where the segmentation results are stored (results for each concentration are stored together)
    output_folder = args.output_folder
    # Path to store individual cell segmentation files (each image result is stored separately)
    output_fen = args.output_fen


    #Original->Label
    os.makedirs(masks_path, exist_ok=True)
    filenames =os.listdir(path)
    #Loading Models
    model = models.Cellpose(gpu=use_GPU, model_type='cyto',nchan=3)
    channels = [1,1]
    for filename in filenames:
      print(f"begin:{filename}")
      files = []
      paths = os.path.join(path,filename)
      #Create a catalog
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

    #Label Segmentation
    for image_files in os.listdir(masks_path):
        output_folder_w=os.path.join(output_folder,image_files)
        os.makedirs(output_folder_w,exist_ok=True)
        mask_files_w = os.path.join(image_folder,image_files)
        image_files_w=os.path.join(masks_path,image_files)
        f_image_files_w=os.path.join(output_fen,image_files)
        os.makedirs(f_image_files_w, exist_ok=True)
        for image_file in os.listdir(image_files_w):
            #Read image
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

            # Calculates the absolute value of the gradient and converts it back to 8 bits
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            # Combined gradient (approximate)
            edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            # Obtain profile information for each cell
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Save the location information of each cell
            cell_positions = []

            for i, contour in enumerate(contours):
                # Get the bounding box for each cell
                x, y, w, h = cv2.boundingRect(contour)
                if w>=10 and h>=10 and w<45 and h<45:
                    # Image cutting based on coordinate information
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

