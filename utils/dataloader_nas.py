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


class catdog_Dataset_nas(Dataset):
	def __init__(self, root_dir, n_superpix=50, mode='train', image_transform=None, label_transform=None):
		self.n_superpix = n_superpix
		self.image_transform = image_transform
		self.label_transform = label_transform

		number_of_image = 0
		if mode == 'train':
			number_of_image = 500
		else:
			number_of_image = 100

		mode_dir = os.path.join(root_dir, mode)
		subject_list = os.listdir(mode_dir)[:number_of_image]
		
		self.image_paths = []
		self.label_paths = []
		self.A_mat_paths = []
		self.superpix_paths = []
		for subject in subject_list:
			subject_path = os.path.join(mode_dir, subject)
			image_path = os.path.join(subject_path, 'img_resize.jpg')
			label_path = os.path.join(subject_path, 'label_resize.png')
			A_mat_path = os.path.join(subject_path, 'A_matrix_500.npy')
			superpix_path = os.path.join(subject_path, 'superpix_500.npy')

			self.image_paths.append(image_path)
			self.label_paths.append(label_path)
			self.A_mat_paths.append(A_mat_path)
			self.superpix_paths.append(superpix_path)


	def __getitem__(self, idx):
		img_path = self.image_paths[idx]
		class_name = img_path.split('/')[-2]
		img_name = os.path.basename(img_path)
		
		image = Image.open(self.image_paths[idx])
		label = Image.open(self.label_paths[idx])
		A_mat = np.load(self.A_mat_paths[idx])
		superpix = np.load(self.superpix_paths[idx])
		superpix_img = mark_boundaries(np.array(image), superpix)
		
		label_np = np.array(label)
		label_np[label_np != 1] = 0
		label = Image.fromarray(label_np*255)
		try:
			if self.image_transform is not None:
				image = self.image_transform(image)
			if self.image_transform is not None:
				label = self.label_transform(label)
		except:
			print(image_numpy.shape)
			print(self.image_paths[idx])
			sys.exit()
		
		return image, label, superpix, superpix_img, img_path

	def __len__(self):
		return len(self.image_paths)

class FSS_Dataset_nas(Dataset):
	def __init__(self, root_dir, n_superpix=50, mode='train', transform=None):
		self.n_superpix = n_superpix
		self.transform = transform

		class_names = sorted(os.listdir(root_dir))
		classes = list(range(0, len(class_names)))
		
		self.image_paths = []
		self.label_paths = []
		cnt = 0

		if mode == 'train':
			start, end = 0, 7
		else:
			start, end = 7, 10
		for class_name in class_names:
			class_path = os.path.join(root_dir, class_name)
			for i in range(start, end):
				image_path = os.path.join(class_path, '{}.jpg'.format(i+1))
				label_path = os.path.join(class_path, '{}.png'.format(i+1))

				self.image_paths.append(image_path)
				self.label_paths.append(label_path)
			if cnt >= 10:
				break
			cnt += 1

		print("total images:", len(self.image_paths))
	def __getitem__(self, idx):
		class_name = self.image_paths[idx].split('/')[-2]
		img_name = '{}_{}'.format(class_name, os.path.basename(self.image_paths[idx]))
		
		image = cv2.imread(self.image_paths[idx])# / 255.0
		# bgr to rgb
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		slic_arr = slic(image, n_segments=self.n_superpix,
								compactness = 10,
								sigma=1.0,
								start_label=0)
		image = np.transpose(image, (2, 0, 1))
		label = cv2.imread(self.label_paths[idx])
		label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
		image = image/255.0
		label = label/255.0


		img = torch.FloatTensor(image)
		label = torch.FloatTensor(label).unsqueeze(0)
		

		return image, label, slic_arr, img_name

	def __len__(self):
		return len(self.image_paths)

if __name__ == '__main__':
	root_dir = '/home/junghwan/workspace/FSS/datasets/few_shot_seg_1000/fewshot_data'
	dataset = FSS_Dataset(root_dir)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
	for image, label in dataloader:
		print('hi')

