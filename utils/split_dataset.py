import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import os
import sys
import argparse
import cv2
from PIL import Image
import shutil
import glob
import random

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

nas_slic_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/preprocessed/slic_seg'
fss_path = '../../datasets/few_shot_seg_1000/fewshot_data'

catdog_path = '../../datasets/catdog'
nas_catdog = make_dir('/media/NAS/nas_187/datasets/junghwan/experience/superpixel/datasets', 'catdog')
# split_data_path = make_dir('../../datasets/', 'simplesplit_catdog')
split_data_path = make_dir('../../datasets/', 'split_catdog')

def Arg():
    parser = argparse.ArgumentParser(description='superpixel segmentation')
    parser.add_argument('-d', '--data_type', dest='data_type', default='catdog',
                        help='set data_type, default is catdog')
    
    return parser.parse_args()

def for_CATDOG():
    image_dir_path = os.path.join(catdog_path, 'images')
    label_dir_path = os.path.join(catdog_path, 'annotations/trimaps')
    graph_dir_path = make_dir(catdog_path, 'for_graph')
    txt_path = os.path.join(catdog_path, 'annotations')
    
    mode_path = {}
    for mode in ['train', 'val', 'test']:
        mode_path[mode] = make_dir(split_data_path, mode)
    mode_cnt = {'train':500, 'val':100, 'test':100}

    mode_name_list = {}
    trainval_txt = open(os.path.join(txt_path, 'trainval.txt'), 'r')
    trainval_list = []
    while True:
        line = trainval_txt.readline()
        if not line: break
        trainval_list.append(line.split(' ')[0])
    trainval_txt.close()

    test_txt = open(os.path.join(txt_path, 'test.txt'), 'r')
    mode_name_list['test'] = []
    while True:
        line = test_txt.readline()
        if not line: break
        mode_name_list['test'].append(line.split(' ')[0])
    test_txt.close()
    random.shuffle(trainval_list)
    mode_name_list['val'] = trainval_list[:300]#[:100]
    mode_name_list['train'] = trainval_list[300:]#[:500]

    n_segment_list = [50, 250, 500]
    # n_segment_list = [50]
    
    for mode in ['train', 'val', 'test']:
    # for mode in ['train']:
        cnt = 0
        new_image_dir_path = make_dir(mode_path[mode], 'images', remove=True)
        new_label_dir_path = make_dir(mode_path[mode], 'labels', remove=True)
        new_graph_dir_path = make_dir(mode_path[mode], 'for_graph', remove=True)
        for n_segment in n_segment_list:
            make_dir(new_graph_dir_path, 'slic_{}'.format(n_segment))
        for image_name in mode_name_list[mode]:
            try:
                for n_segment in n_segment_list:
                    n_seg_path = os.path.join(graph_dir_path, 'slic_{}'.format(n_segment))
                    supix_img_path = os.path.join(n_seg_path, image_name+'.jpg')
                    adj_path = os.path.join(n_seg_path, image_name+'_adj.npy')
                    slic_path = os.path.join(n_seg_path, image_name+'_slic.npy')
                    graph_label_path = os.path.join(n_seg_path, image_name+'_graphlabel.npy')
                    adj_label_path = os.path.join(n_seg_path, image_name+'_adj_label.npy')

                    new_n_seg_path = os.path.join(new_graph_dir_path, 'slic_{}'.format(n_segment))
                    new_supix_img_path = os.path.join(new_n_seg_path, image_name+'.jpg')
                    new_adj_path = os.path.join(new_n_seg_path, image_name+'_adj.npy')
                    new_slic_path = os.path.join(new_n_seg_path, image_name+'_slic.npy')
                    new_graph_label_path = os.path.join(new_n_seg_path, image_name+'_graphlabel.npy')
                    new_adj_label_path = os.path.join(new_n_seg_path, image_name+'_adj_label.npy')

                    shutil.copy(supix_img_path, new_supix_img_path)
                    shutil.copy(adj_path, new_adj_path)
                    shutil.copy(slic_path, new_slic_path)
                    shutil.copy(graph_label_path, new_graph_label_path)
                    shutil.copy(adj_label_path, new_adj_label_path)
                image_path = os.path.join(image_dir_path, image_name+'.jpg')
                label_path = os.path.join(label_dir_path, image_name+'.png')

                new_image_path = os.path.join(new_image_dir_path, image_name+'.jpg')
                new_label_path = os.path.join(new_label_dir_path, image_name+'.png')

                shutil.copy(image_path, new_image_dir_path)
                shutil.copy(label_path, new_label_path)
                cnt += 1
            except:
                print(image_name)
            # if cnt == mode_cnt[mode]:
            #     break

        

def main():
    for_CATDOG()
    # for_FSS()

if __name__ == '__main__':
    main()
