import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataloader import catdog_Dataset
from utils.build_graph import gen_graph
from utils.gen_graph_label import gen_graph_label
from utils.utils import *
from test_networks.test_gnn import test_gnn
from test_networks.test_gat import test_gat
from test_networks.test_unet import test_unet
from test_networks.test_gunet import test_gunet

from networks.encoder import Encoder
from networks.decoder import Decoder
from graph_networks.models import GAT, GCN, GNN, GraphUnet

import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
		rows, cols = np.where(slic_arr==1)
		mask = (slic_arr == i)*(val.data.cpu())
		predict_img += mask.numpy()

	return predict_img



def Arg():
	parser = argparse.ArgumentParser(description='superpixel segmentation')
	parser.add_argument('-n', '--network', dest='network', default='gcn',
						help='set network type, unet, gnn, gcn, gat, default is gcn')
	parser.add_argument('-d', '--dataset', dest='dataset', default='fss',
						help='set dataset, fss or catdog, default is fss')
	parser.add_argument('-g', '--gpu_id', dest='gpu_id', default=0, type=int,
						help='set gpu id, 0~7, default is 0')
	parser.add_argument('-e', '--epochs', dest='epochs', default=10, type=int,
						help='set number of epoch, default is 10')

	parser.add_argument('--superpix_number', dest='superpix_number', default=50, type=int,
						help='set number of superpix, default is 50')
	parser.add_argument('--n_layer', dest='n_layer', default=2, type=int,
						help='set number of layers, 1~5, default is 2')
	parser.add_argument('--n_hid', dest='n_hid', default=16, type=int,
						help='set number of hidden layer, 1~5, default is 16')
	parser.add_argument('--encoder_parameter_fix', dest='encoder_parameter_fix', default=False, type=bool,
						help='set encoder parameter fix, True or False, default is False')
	parser.add_argument('--concat', dest='concat', default=False, type=bool,
						help='set concat mode, True or False, default is False')
	parser.add_argument('--use_gnn_encoder', dest='use_gnn_encoder', default=False, type=bool,
						help='set use_gnn_encoder mode, True or False, default is False')
	parser.add_argument('--use_gnn_parameter', dest='use_gnn_parameter', default=False, type=bool,
						help='set use_gnn_parameter mode, True or False, default is False')
	parser.add_argument('--multi_try', dest='multi_try', default=1, type=int,
						help='set multi_try mode, 1~10, default is 1')
	parser.add_argument('--data_root', dest='data_root', default='local',
						help='set data_root mode, local or nas, default is local')
	parser.add_argument('--attention_head', dest='attention_head', default=3, type=int,
						help='set attention_head, default is 3')
	
	parser.add_argument('--graph_feature', dest='graph_feature', default='encoder',
						help='set type of graph featrue, encoder or decoder or None, default is encoder')

	return parser.parse_args()

def main():
	args = Arg()
	global device
	device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

	nas_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/results/{}'.format(args.dataset)
	nas_model_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/models'
	if args.network == 'unet':
		nas_dir = make_dir(nas_dir, 'unet_{}'.format(args.data_root))
		nas_dir = make_dir(nas_dir, 'unet_encoderfix:{}'.format(args.encoder_parameter_fix))
	else:
		nas_dir = make_dir(nas_dir, '{}_{}'.format(args.network, args.data_root))
		nas_dir = make_dir(nas_dir, '{}_supix:{}_nhid:{}_nlayer:{}_encoderfix:{}_concat:{}_gnnencoder:{}_gnnparameter:{}_head:{}_gft:{}_epochs:{}'.format(args.network, 
			args.superpix_number, args.n_hid, args.n_layer, args.encoder_parameter_fix, args.concat, args.use_gnn_encoder, args.use_gnn_parameter, args.attention_head,
			args.graph_feature, args.epochs))
	if args.multi_try > 1:
		nas_dir = make_dir(nas_dir, 'multi_try')
	print('result dir')
	print(nas_dir)

	image_transforms = {
		'train': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
	label_transforms = {
		'train': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		]),
		'test': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		]),
	}

	if args.dataset == 'fss':
		FSS_dir = '../datasets/few_shot_seg_1000/fewshot_data'
		dataset = FSS_Dataset(FSS_dir, n_superpix=args.superpix_number, mode='test', image_transform=image_transforms['test'], label_transform=label_transforms['test'])
	elif args.dataset == 'catdog':
		# catdog_dir = '../datasets/catdog'
		catdog_dir = '../datasets/split_catdog'
		dataset = catdog_Dataset(catdog_dir, net=args.network, n_superpix=args.superpix_number, mode='test', image_transform=image_transforms['test'], label_transform=label_transforms['test'])
	elif args.dataset == 'simplecatdog':
		# catdog_dir = '../datasets/catdog'
		catdog_dir = '../datasets/simplesplit_catdog'
		dataset = catdog_Dataset(catdog_dir, net=args.network, n_superpix=args.superpix_number, mode='test', image_transform=image_transforms['test'], label_transform=label_transforms['test'])
# catdog_Dataset(catdog_dir, n_superpix=args.superpix_number, mode=x, image_transform=image_transforms[x], label_transform=label_transforms[x])
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

	save_root = os.path.join(nas_model_dir, args.data_root, args.network+'_epoch:{}'.format(args.epochs), args.dataset)
	if args.multi_try > 1:
		save_root = make_dir(save_root, 'multi_try')
		result_txt = os.path.join(nas_dir, 'total_result.txt')
		f = open(result_txt, 'w')

	pretrained_path = './pretrained_model/vgg16-397923af.pth'

	whole_dsc = 0.0
	whole_loss = 0.0
	for i in range(args.multi_try):
		if args.network == 'unet':
			save_path = make_dir(save_root, 'encoderfix:{}_iter:{}'.format(args.encoder_parameter_fix, i))
			encoder_path = os.path.join(save_path, 'encoder_encoderfix:{}_{}.pth'.format(args.encoder_parameter_fix, i))
			decoder_path = os.path.join(save_path, 'decoder_{}_{}.pth'.format(args.encoder_parameter_fix, i))
			encoder = Encoder(pretrained_path, device, args.network)
			decoder = Decoder().to(device)

			encoder.load_state_dict(torch.load(encoder_path))
			decoder.load_state_dict(torch.load(decoder_path))

			criterion = nn.BCELoss()

			total_dsc, total_loss = test_unet(encoder, decoder ,dataloader, criterion, nas_dir, device, i)

		else:
			save_path = make_dir(save_root, 'superpix:{}_nhid:{}_nlayer:{}_encoderfix:{}_concat:{}_gnnencoder:{}_gnnparameter:{}_head:{}_gft:{}_iter:{}'.format(args.superpix_number, 
				args.n_hid, args.n_layer, args.encoder_parameter_fix, args.concat, args.use_gnn_encoder, args.use_gnn_parameter, args.attention_head, args.graph_feature, i))
			print('model path')
			print(save_path)
			encoder_path = os.path.join(save_path, 'encoder.pth')
			encoder = Encoder(pretrained_path, device, args.network)

			gnn_path = os.path.join(save_path, '{}.pth'.format(args.network))
			if args.network == 'gnn':
				gnn = GNN(nfeat=512, nhid=args.n_hid, nclass=1, dropout=0.5, n_layer=args.n_layer, concat=args.concat).to(device)
			elif args.network == 'gcn':
				gnn = GCN(nfeat=512, nhid=args.n_hid, nclass=1, dropout=0.5, n_layer=args.n_layer, concat=args.concat).to(device)
			elif 'gat' in args.network:
				gnn = GAT(nfeat=512, nhid=args.n_hid, nclass=1, dropout=0.5, nheads=args.attention_head, alpha=0.2, n_layer=args.n_layer, concat=args.concat, gatType=args.network).to(device)
			elif 'gunet' in args.network:
				gnn = GraphUnet(nfeat=512, nhid=args.n_hid, nclass=1, dropout=0.5, alpha=0.2, n_layer=args.n_layer, concat=args.concat).to(device)

			encoder.load_state_dict(torch.load(encoder_path, map_location='cuda:0'))
			gnn.load_state_dict(torch.load(gnn_path, map_location='cuda:0'))
			criterion = nn.BCELoss()

			if 'gunet' in args.network:
				# print('gunet start!')
				total_dsc, total_loss = test_gunet(encoder, gnn, dataloader, criterion, nas_dir, device, i, concat=args.concat, network=args.network)
			elif 'gat' in args.network:
				total_dsc, total_loss = test_gat(encoder, gnn, dataloader, criterion, nas_dir, device, i, concat=args.concat, network=args.network, graph_feature=args.graph_feature)
			else:
				total_dsc, total_loss = test_gnn(encoder, gnn, dataloader, criterion, nas_dir, device, i, concat=args.concat)

		if args.multi_try > 1:
			whole_dsc += total_dsc
			whole_loss += total_loss
			f.write('[{}] dsc: {:.4f}'.format(i, whole_dsc/args.multi_try))

	if args.multi_try > 1:
		f.write('all average dsc: {:.4f}'.format(whole_dsc/args.multi_try))

if __name__ == '__main__':
	main()
