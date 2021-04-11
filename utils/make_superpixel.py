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
# from utils import *

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
# catdog_path = '../../datasets/catdog/images'
# np.set_printoptions(threshold=sys.maxsize)
city_nas_path = '/media/NAS/nas_187/datasets/cityscapes'
city_path = '../../datasets/cityscapes'
city_jh_path = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/datasets/cityscapes'

def Arg():
	parser = argparse.ArgumentParser(description='superpixel segmentation')
	parser.add_argument('-d', '--data_type', dest='data_type', default='catdog',
						help='set data_type, default is catdog')

	return parser.parse_args()

def adj_normalize(adj):
	rowsum = np.array(adj.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	adj = r_mat_inv.dot(adj)
	return adj

def gen_adj_label(adj, graph_label):
	x, y = adj.shape
	adj_label = adj.copy()
	for i in range(x):
		for j in range(y):
			if graph_label[i] != graph_label[j]:
				adj_label[i, j] = 0

	return adj_label

def gen_adj(slic_arr):
	n_superpixel = np.unique(slic_arr)
	image_size = slic_arr.shape
	x, y = slic_arr.shape
	adj = np.zeros((len(n_superpixel), len(n_superpixel)), dtype=np.uint8)
	for i in range(x-1):
		for j in range(y-1):
			ff_area = slic_arr[i:i+2, j:j+2]
			# print(ff_area)
			# print(ff_area.shape)
			# sys.exit()

			if ff_area[0, 0] != ff_area[0, 1]:
				a = ff_area[0, 0]
				b = ff_area[0, 1]
				adj[a, b] = 1

			if ff_area[0, 0] != ff_area[1, 0]:
				a = ff_area[0, 0]
				b = ff_area[1, 0]
				adj[a, b] = 1

			if ff_area[0, 0] != ff_area[1, 1]:
				a = ff_area[0, 0]
				b = ff_area[1, 1]
				adj[a, b] = 1

			if ff_area[0, 1] != ff_area[1, 0]:
				a = ff_area[0, 1]
				b = ff_area[1, 0]
				adj[a, b] = 1

			if ff_area[0, 1] != ff_area[1, 1]:
				a = ff_area[0, 1]
				b = ff_area[1, 1]
				adj[a, b] = 1

			if ff_area[1, 0] != ff_area[1, 1]:
				a = ff_area[1, 0]
				b = ff_area[1, 1]
				adj[a, b] = 1

	# adj = adj_normalize(adj + sp.eye(adj.shape[0]))
	# adj = torch.FloatTensor(adj)
	return adj

def gen_graph_label(label, slic_arr):
	# label = torch.squeeze(label)

	graph_label = []
	for superpixel_val in np.unique(slic_arr):
		mask = slic_arr == superpixel_val
		temp_map = label[mask]
		node = 1.0 if temp_map.mean() > 0.4 else 0.0
		graph_label.append(node)
	graph_label = np.array(graph_label)
	# graph_label = torch.from_numpy(graph_label).unsqueeze(1)
	
	return graph_label

def reconstruct_predict(output, slic_arr):
	predict_img = np.zeros((slic_arr.shape))

	for i, val in enumerate(output):
		mask = (slic_arr == i)*(val)
		predict_img += mask#.numpy()

	return predict_img

def for_CATDOG():
	catdog_img_path = os.path.join(catdog_path, 'images')
	image_names = os.listdir(catdog_img_path)
	image_list = glob.glob(catdog_path + '/images/*.jpg')
	label_dir = os.path.join(catdog_path, 'annotations', 'trimaps')
	graph_path = make_dir(catdog_path, 'for_graph', remove=False)

	n_segment_list = [1000]
	for n_segment in n_segment_list:
		print('now : ', n_segment)
		n_seg_path = make_dir(graph_path, 'slic_{}_label'.format(n_segment))
		cnt = 0
		for image_path in image_list:
			image_name = os.path.basename(image_path)
			label_path = os.path.join(label_dir, image_name.replace('.jpg', '.png'))
			# image_path = os.path.join(catdog_img_path, image_name)
			try:
				image = Image.open(image_path)
				label = Image.open(label_path)
				# label.show()
				label_np = np.array(label)#*255
				
				image_np = np.array(image)

				image_np = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)
				label_np = cv2.resize(label_np, (224, 224), interpolation=cv2.INTER_AREA)
				label_np[label_np != 1] = 0
				
				# make slic seg
				slic_seg = slic(image_np, n_segments=n_segment,
									compactness=10,
									sigma=1.0,
									start_label=0)
				slic_seg_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_slic.npy'))

				# make graph label
				graph_label = gen_graph_label(label_np, slic_seg)
				graph_label_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_graphlabel.npy'))				
				graph_img = reconstruct_predict(graph_label, slic_seg)
				graph_img_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_graphimg.png'))
				label_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_label.png'))
				plt.imsave(graph_img_path, graph_img, cmap='gray')
				plt.imsave(label_path, label_np, cmap='gray')
				cnt += 1

			# slic_img = mark_boundaries(image_np, slic_seg)
			# slic_img_path = os.path.join(n_seg_path, image_name)

			# fig, ax = plt.subplots(1, 4)
			# ax[0].imshow(image_np)
			# ax[1].imshow(slic_img)
			# ax[2].imshow(label_np, cmap='gray')
			# ax[3].imshow(graph_img, cmap='gray')

			# plt.show()
			# sys.exit()

			# make adj
			# adj = gen_adj(slic_seg)
			# adj_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_adj.npy'))

			# make adj_label
			# adj_label = gen_adj_label(adj, graph_label)
			# adj_label_path = os.path.join(n_seg_path, image_name.replace('.jpg', '_adj_label.npy'))
			

			

			# np.set_printoptions(threshold=sys.maxsize)
			# plt.imshow(slic_img)
			# plt.imshow(slic_seg)
			# show_graph_with_labels(adj, slic_seg)
			# sys.exit()

			# np.save(adj_path, adj)
			# np.save(slic_seg_path, slic_seg)
			# np.save(graph_label_path, graph_label)
			# np.save(adj_label_path, adj_label)
			# plt.imsave(slic_img_path, slic_img)
			except:
				print('error image, path is : {}'.format(image_path))
			if cnt == 20:
				break


def main():
	for_CATDOG()
	# for_city()
	# for_FSS()

if __name__ == '__main__':
	main()