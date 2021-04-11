import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import cv2
import shutil
import glob
from PIL import Image

# https://scikit-image.org/docs/dev/api/skimage.data.html
# astronaut : a person who is trained to travel in a spacecraft
from skimage.data import astronaut
from skimage.color import rgb2gray

# https://scikit-image.org/docs/dev/api/skimage.filters.html
# Find edges in an image using the soble filter.
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def make_dir(root_dir, target_dir, remove=False):
	result_dir = os.path.join(root_dir, target_dir)
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)
	elif remove == True:
		shutil.rmtree(result_dir)
		os.mkdir(result_dir)

	return result_dir

nas_result_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/results/catdog'
save_dir = make_dir(nas_result_dir, 'aggregation_reseult')
save_dir = make_dir(save_dir, 'best_reseult', remove=True)

model_list = ['unet', 'gnn', 'gcn', 'gat'] # 10, 4, 5, 4 is best
best_iter = [2, 4, 5, 4]
def for_CATDOG():
    model_result_path = {}
    for i, model_name in enumerate(model_list):
        if model_name == 'gnn':
            result_path = os.path.join(nas_result_dir, '{}_local'.format(model_name), 'gnn_500')
        elif model_name == 'gcn':
            result_path = os.path.join(nas_result_dir, '{}_local'.format(model_name), 'gcn_500_ngcn:2')
            # print(result_path)
        elif model_name == 'gat':
            result_path = os.path.join(nas_result_dir, '{}_local'.format(model_name), 'gat_500_ngat:2')
        else:
            result_path = os.path.join(nas_result_dir, '{}_local2'.format(model_name))
        # print(result_path)
        
        if model_name == 'unet':
            multi_path = os.path.join(result_path, 'multi_try')
            result_path = os.path.join(multi_path, 'iter_{}'.format(best_iter[i]-1))
            best_iter_path = os.path.join(result_path, 'result')
        else:
            multi_path = os.path.join(result_path, 'multi_try')
            result_path = os.path.join(multi_path, 'result')
            best_iter_path = os.path.join(result_path, 'iter_{}'.format(best_iter[i]-1))
        

        reuslt_paths = sorted(glob.glob(best_iter_path + '/*.png'))
        model_result_path[model_name] = reuslt_paths

    # print(model_result_path['unet'][80])
    # print(model_result_path['gnn'][80])
    # print(model_result_path['gcn'][80])
    # print(model_result_path['gat'][80])

    for i in range(len(model_result_path['unet'])):
        img = {}
        prediction = {}
        name = os.path.basename(model_result_path['unet'][i])[:-15]+'.png'
        for model_name in model_list:
            img[model_name] = cv2.cvtColor(cv2.imread(model_result_path[model_name][i]),  cv2.COLOR_BGR2RGB)
            # print(model_name, img[model_name].shape)
            prediction[model_name] = img[model_name][:,224:448,:]
        # original_img = np.zeros((224, 224, 3))
        # label = np.zeros((224, 224, 3))
        original_img = img['unet'][:, :224, :]
        label_img = img['gat'][:, 448:, :]
        for model_name in model_list:
            original_img = np.concatenate((original_img, prediction[model_name]), axis=1)
        original_img = np.concatenate((original_img, label_img), axis=1)
        plt.imsave(os.path.join(save_dir, name), original_img)
        
        # print(original_img.shape)
        # print(label_img.shape)
        # sys.exit()

        


        

def main():
    for_CATDOG()
    # for_FSS()

if __name__ == '__main__':
    main()