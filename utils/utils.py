import os
import numpy as np
import shutil
import torch
import networkx as nx
from train_networks.dice import dice_score
import matplotlib.pyplot as plt

def make_dir(root_dir, target_dir, remove=False):
	result_dir = os.path.join(root_dir, target_dir)
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)
	elif remove == True:
		shutil.rmtree(result_dir)
		os.mkdir(result_dir)

	return result_dir

def tensor_to_img(image):
	image = image.data[0].cpu().numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * image + mean
	inp = np.clip(inp, 0, 1)

	return inp

def reconstruct_predict(output, slic_arr):
	predict_img = np.zeros((slic_arr.shape))

	for i, val in enumerate(output):
		mask = (slic_arr == i)*(val.data.cpu())
		predict_img += mask.numpy()

	return predict_img

def overlay_img(img, label):
	img = img[0]
	x, y, c = img.shape
	label_np = np.zeros((x, y ,c))
	label_np[:,:,0] = label

	result = img*0.5 +  label_np*0.5

	return result

def gray_to_rgb(img):
	x, y = img.shape
	empty = np.zeros((x, y, 3))
	empty[:,:,0] = img
	empty[:,:,1] = img
	empty[:,:,2] = img

	return empty

def save_predict_img(image, label, output, slic_arr, superpix_img, nas_dir, img_name):
	predict_img = reconstruct_predict(output, slic_arr)
	predict_img[predict_img>=0.5] = 1
	predict_img[predict_img<0.5] = 0
	img_np = tensor_to_img(image)#.cpu().numpy()
	label_np = label.data[0].cpu().numpy()[0]

	overlay_predict = overlay_img(superpix_img, predict_img).numpy()
	overlay_label = overlay_img(superpix_img, label_np).numpy()
	
	concat_img = np.concatenate((img_np, overlay_predict, overlay_label), axis=1)
	dsc = dice_score(predict_img, label_np)
	save_path = os.path.join(nas_dir, img_name.replace('.jpg', '_dsc:{:.4f}.png'.format(dsc)))
	
	plt.imsave(save_path, concat_img, cmap='gray')

	return dsc

def save_unet_img(image, label, output, slic_arr, superpix_img, nas_dir, img_name):
	# predict_img = reconstruct_predict(output, slic_arr)
	predict_img = output.cpu().numpy()
	predict_img[predict_img>=0.5] = 1
	predict_img[predict_img<0.5] = 0
	img_np = tensor_to_img(image)#.cpu().numpy()
	label_np = label.data[0].cpu().numpy()[0]

	overlay_predict = overlay_img(superpix_img, predict_img).numpy()
	overlay_label = overlay_img(superpix_img, label_np).numpy()
	
	concat_img = np.concatenate((img_np, overlay_predict, overlay_label), axis=1)
	dsc = dice_score(predict_img, label_np)
	save_path = os.path.join(nas_dir, img_name.replace('.jpg', '_dsc:{:.4f}.png'.format(dsc)))
	
	plt.imsave(save_path, concat_img, cmap='gray')

	return dsc

# for saving featuremaps
def normalize_output(img):
	# plus_img = np.zeros_like(img)
	# plus_img[img>0] = img[img>0]
	# # print('plus_img min:', plus_img.min())
	# plus_img = plus_img-plus_img.min()

	# plus_img = plus_img/(plus_img.max() + 0.0001)

	# minus_img = np.zeros_like(img)
	# minus_img[img<0] = img[img<0]
	# # print('minus_img min:', minus_img.min())
	# minus_img = minus_img-minus_img.min()
	# minus_img = minus_img/(minus_img.max() + 0.0001)

	# return_img = np.zeros_like(img)
	# return_img[img>0] = plus_img[img>0]+1
	# # img = img + plus_img+1
	# return_img = return_img + minus_img
	# return return_img
	img = img-img.min()
	img = img/(img.max())
	return img
	

def show_featuremaps(ft_interpolated, slic_arr, nas_dir, img_name, feature_name):
	for_testing = make_dir(nas_dir, 'test_feature')
	img_test_dir = make_dir(for_testing, img_name.replace('.jpg', ''))
	feature_name_dir = make_dir(img_test_dir, feature_name)

	for i, feature_map in enumerate(ft_interpolated[0]):
		feature_path = os.path.join(feature_name_dir, '{}.png'.format(i))
		plt.imsave(feature_path, normalize_output(feature_map.cpu().numpy()), cmap='gray')

def save_original_img(image, label, superpix_img, nas_dir, img_name):
	for_testing = make_dir(nas_dir, 'test_feature', remove=False)
	img_test_dir = make_dir(for_testing, img_name.replace('.jpg', ''))

	label_path = os.path.join(img_test_dir, 'label.png')
	plt.imsave(label_path, label[0].cpu()[0], cmap='gray')
	img_path = os.path.join(img_test_dir, 'img.jpg')
	img_np = tensor_to_img(image)
	plt.imsave(img_path, img_np)
	boundary_pth = os.path.join(img_test_dir, 'boundary.jpg')
	boundary_img = superpix_img
	# print(superpix_img.shape)
	plt.imsave(boundary_pth, superpix_img[0].numpy())

def show_graph_with_labels(adj, slic_arr, idx=None):
	n, _ = adj.shape

	node_position = {}
	for i in range(n):
		rows, cols = np.where(slic_arr==i)
		y_center = (rows.mean())# / 2
		x_center = (cols.mean())# / 2
		# node_position[i] = (y_center, x_center)
		node_position[i] = (x_center, y_center)
	
	# print('n:', n)
	rows, cols = np.where(adj==1)
	G = nx.Graph()
	for i in range(n):
		G.add_node(i, pos=node_position[i])

	edge_label = {}
	for i in range(n):
		# cols = np.where(adj[i] == 1)
		for j in range(n):
			if adj[i, j] > 0:
				# print(f'adding edge {i}, {j}')
				G.add_edge(i, j)
				if i == idx: # N-250:110, 87, 69, 31, 27 / N-500:174, 186 / N-50:14, 31
					edge_label[(i, j)] = '{:.2f}'.format(adj[i, j])
	# print(edge_label)
	# print(adj[idx])
	pos = nx.get_node_attributes(G, 'pos')
	nx.draw_networkx_nodes(G, pos, node_size=1)
	# nx.draw_networkx_labels(G, pos, font_size=8)
	nx.draw_networkx_edges(G, pos, edge_color='r')
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=7, font_color='red')
	plt.show()