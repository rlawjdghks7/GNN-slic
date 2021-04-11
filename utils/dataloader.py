import torch
from torch.utils.data import Dataset
import torchvision
import os
import sys
import glob
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sp

class catdog_Dataset(Dataset):
	def __init__(self, root_dir, net='unet', feature='encoder', n_superpix=50, mode='train', image_transform=None, label_transform=None,
				build_feature=False):
		self.n_superpix = n_superpix
		self.mode = mode
		self.image_transform = image_transform
		self.label_transform = label_transform
		self.net = net
		self.feature = feature
		self.build_feature = build_feature

		mode_dir = os.path.join(root_dir, mode)
		image_dir = os.path.join(mode_dir, 'images')
		label_dir = os.path.join(mode_dir, 'labels')
		graph_dir = os.path.join(mode_dir, 'for_graph')
		slic_dir = os.path.join(graph_dir, 'slic_{}'.format(n_superpix))
		supix_img_paths = glob.glob(slic_dir + '/*.jpg')
		if build_feature == False:
			feature_graph_dir = os.path.join(graph_dir, '{}_slic_{}'.format(feature, n_superpix))

		self.image_paths = []
		self.label_paths = []
		self.adj_paths = []
		self.slic_paths = []
		self.supix_paths = []
		self.graph_label_paths = []
		self.adj_label_paths = []
		self.featrue_graph_paths = []

		for supix_img_path in supix_img_paths:
			image_name = os.path.basename(supix_img_path)

			image_path = os.path.join(image_dir, image_name)
			label_path = os.path.join(label_dir, image_name.replace('.jpg', '.png'))
			adj_path = os.path.join(slic_dir, image_name.replace('.jpg', '_adj.npy'))
			slic_path = os.path.join(slic_dir, image_name.replace('.jpg', '_slic.npy'))
			adj_label_path = os.path.join(slic_dir, image_name.replace('.jpg', '_adj_label.npy'))
			graph_label_path = os.path.join(slic_dir, image_name.replace('.jpg', '_graphlabel.npy'))
			if build_feature == False:
				feature_graph_path = os.path.join(feature_graph_dir, image_name.replace('.jpg', '_{}_feature.npy'.format(feature)))

			self.image_paths.append(image_path)
			self.label_paths.append(label_path)
			self.adj_paths.append(adj_path)
			self.slic_paths.append(slic_path)
			self.supix_paths.append(supix_img_path)
			self.graph_label_paths.append(graph_label_path)
			self.adj_label_paths.append(adj_label_path)
			self.featrue_graph_paths.append(feature_graph_path)

	def __getitem__(self, idx):
		img_path = self.image_paths[idx]
		# img_name = os.path.basename(img_path)
		
		image = Image.open(self.image_paths[idx])#.convert('RGB')
		label = Image.open(self.label_paths[idx])#.convert('L')
		label_np = np.array(label)
		label_np[label_np != 1] = 0
		label = Image.fromarray(label_np*255)
		
		supix_img = Image.open(self.supix_paths[idx])
		supix_img = np.array(supix_img)/255

		slic_arr = np.load(self.slic_paths[idx])
		slic_arr = torch.FloatTensor(slic_arr)
		supix_img = torch.FloatTensor(supix_img)

		if not self.net == 'unet':
			adj_arr = np.load(self.adj_paths[idx])			
			graph_label_arr = np.load(self.graph_label_paths[idx])
			adj_label_arr = np.load(self.adj_label_paths[idx])
			feature_graph_arr = np.load(self.featrue_graph_paths[idx])
		# try:
		# 	if self.image_transform is not None:
		# 		image = self.image_transform(image)
		# 	if self.image_transform is not None:
		# 		label = self.label_transform(label)
		# except:
		# 	print(self.image_paths[idx])
		# print(image)
		if self.image_transform is not None:
			image = self.image_transform(image)
		if self.image_transform is not None:
			label = self.label_transform(label)

		if self.net == 'unet':
			slic_arr = torch.FloatTensor(slic_arr)
			supix_img = torch.FloatTensor(supix_img)

			return image, label, supix_img, slic_arr, img_path

		else:
			adj_arr = self.adj_normalize(adj_arr + sp.eye(adj_arr.shape[0]))
			adj = torch.FloatTensor(adj_arr)
			
			graph_label = torch.FloatTensor(graph_label_arr).unsqueeze(1)
			np.set_printoptions(threshold=sys.maxsize)
			adj_label_arr = self.adj_normalize(adj_label_arr + sp.eye(adj_label_arr.shape[0]))
			adj_label = torch.FloatTensor(adj_label_arr)
			feature_graph = torch.FloatTensor(feature_graph_arr)			
			
			return image, label, feature_graph, adj, slic_arr, supix_img, graph_label, adj_label, img_path
	
	def adj_normalize(self, adj):
		rowsum = np.array(adj.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = sp.diags(r_inv)
		adj = r_mat_inv.dot(adj)
		return adj

	def __len__(self):
		return len(self.image_paths)