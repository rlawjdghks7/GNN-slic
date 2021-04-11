import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataloader import FSS_Dataset, catdog_Dataset
from utils.dataloader_nas import FSS_Dataset_nas, catdog_Dataset_nas
from utils.build_graph import gen_graph
from utils.gen_graph_label import gen_graph_label
from utils.utils import *
from train_networks.train_unet import train_unet
from train_networks.train_gnn import train_gnn
from train_networks.train_gat import train_gat

from networks.encoder import Encoder
from networks.decoder import Decoder
from graph_networks.models import GAT, GCN, GNN

import os
import sys
import numpy as np
import random
from tqdm import tqdm
import shutil
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(1)
# random.seed(1)

# torch.manual_seed(1050879)
# torch.cuda.manual_seed(1050879)
# torch.cuda.manual_seed_all(1050879)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(1050879)
# random.seed(1050879)


def Arg():
	parser = argparse.ArgumentParser(description='superpixel segmentation')
	parser.add_argument('-n', '--network', dest='network', default='gcn',
						help='set network type, unet, gnn, gcn, gat, default is gcn')
	parser.add_argument('-d', '--dataset', dest='dataset', default='fss',
						help='set dataset, fss or catdog, simplecatdog, default is fss')
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
	parser.add_argument('--encoder_parameter_fix', dest='encoder_parameter_fix', default=False, type=bool, # action='store_true', 
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
	
	return parser.parse_args()

def main():
	args = Arg()
	global device
	device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

	nas_dir = make_dir('/media/NAS/nas_187/datasets/junghwan/experience/superpixel/results', format(args.dataset))
	nas_model_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/models'
	if args.network == 'unet':
		nas_dir = make_dir(nas_dir, 'unet_{}'.format(args.data_root))
		nas_dir = make_dir(nas_dir, 'unet_encoderfix:{}'.format(args.encoder_parameter_fix))
	else:
		nas_dir = make_dir(nas_dir, '{}_{}'.format(args.network, args.data_root))
		nas_dir = make_dir(nas_dir, '{}_supix:{}_nhid:{}_nlayer:{}_encoderfix:{}_concat:{}_gnnencoder:{}_gnnparameter:{}_head:{}'.format(args.network, 
			args.superpix_number, args.n_hid, args.n_layer, args.encoder_parameter_fix, args.concat, args.use_gnn_encoder, args.use_gnn_parameter, args.attention_head))
	if args.multi_try > 1:
		nas_dir = make_dir(nas_dir, 'multi_try')
	
	image_transforms = {
		'train': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
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
		'val': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		]),
	}
	n_class = 1
	if args.dataset == 'fss':
		if args.data_root == 'local':
			FSS_dir = '../datasets/few_shot_seg_1000/fewshot_data'
			# dataset = {x: FSS_Dataset(FSS_dir, mode=x, image_transform=image_transforms[x], label_transform=label_transforms[x]) for x in ['train', 'test']}
			dataset = {x: FSS_Dataset(FSS_dir, n_superpix=args.superpix_number, mode=x) for x in ['train', 'test']}
	elif args.dataset == 'catdog':
		if args.data_root == 'local':
			# catdog_dir = '../datasets/catdog'
			catdog_dir = '../datasets/split_catdog'
			# catdog_dir = '../datasets/simplesplit_catdog'
			dataset = {x: catdog_Dataset(catdog_dir, n_superpix=args.superpix_number, mode=x, image_transform=image_transforms[x], label_transform=label_transforms[x]) for x in ['train', 'val']}
		elif args.data_root == 'nas':
			catdog_dir = '/media/NAS/nas_187/soopil/data/catdog_superpix'
	elif args.dataset == 'simplecatdog':
		catdog_dir = '../datasets/simplesplit_catdog'
		dataset = {x: catdog_Dataset(catdog_dir, n_superpix=args.superpix_number, mode=x, image_transform=image_transforms[x], label_transform=label_transforms[x]) for x in ['train', 'val']}
	elif args.dataset == 'city':
		city_dir = '/media/NAS/nas_187/datasets/junghwan/experience/superpixel/datasets/cityscapes/for_graph_resize'
		dataset = {x: city_Dataset(city_dir, n_superpix=args.superpix_number, mode=x, image_transform=image_transforms[x], label_transform=label_transforms[x]) for x in ['train', 'val']}
		n_class = 19
	dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=1, shuffle=True, num_workers=4) for x in ['train', 'val']}

	pretrained_path = './pretrained_model/vgg16-397923af.pth'

	# save_root = os.path.join('models', args.data_root, args.network, args.dataset)
	save_root = make_dir(nas_model_dir, args.data_root)
	save_root = make_dir(save_root, args.network)
	save_root = make_dir(save_root, args.dataset)
	if args.multi_try > 1:
		save_root = make_dir(save_root, 'multi_try')

	for i in range(args.multi_try):
		if args.network == 'unet':
			save_path = make_dir(save_root, 'encoderfix:{}_iter:{}'.format(args.encoder_parameter_fix, i))
			encoder = Encoder(pretrained_path, device, args.network, parameter_fix=args.encoder_parameter_fix)
			decoder = Decoder(output_channel=n_class).to(device)

			optimizer = optim.Adam(list(encoder.parameters()) +
									list(decoder.parameters()),
									lr=0.01, weight_decay=5e-4)
			criterion = nn.BCELoss()

			encoder, decoder = train_unet(encoder, decoder, dataloader, optimizer, criterion, nas_dir, device, i, epochs=args.epochs)

			torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder_encoderfix:{}_{}.pth'.format(args.encoder_parameter_fix, i)))
			torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder_{}_{}.pth'.format(args.encoder_parameter_fix, i)))
		else:
			save_path = make_dir(save_root, 'superpix:{}_nhid:{}_nlayer:{}_encoderfix:{}_concat:{}_gnnencoder:{}_gnnparameter:{}_head:{}_iter:{}'.format(args.superpix_number, 
				args.n_hid, args.n_layer, args.encoder_parameter_fix, args.concat, args.use_gnn_encoder, args.use_gnn_parameter, args.attention_head, i))
			encoder_path = os.path.join(save_path, 'encoder.pth'.format(args.superpix_number, args.encoder_parameter_fix, args.concat, i))
			gnn_path = os.path.join(save_path, '{}.pth'.format(args.network))

			encoder = Encoder(pretrained_path, device, args.network, parameter_fix=args.encoder_parameter_fix)
			if args.use_gnn_encoder:
				gnn_encoder_path = os.path.join(save_root.replace(args.network, 'gnn'), 'superpix:{}_nhid:{}_nlayer:{}_encoderfix:{}_concat:{}_gnnencoder:{}_gnnparameter:{}_iter:{}'.format(args.superpix_number, 
				args.n_hid, args.n_layer, args.encoder_parameter_fix, args.concat, False, False, i), 'encoder.pth')
				encoder.load_state_dict(torch.load(gnn_encoder_path))

			if args.network == 'gnn':
				gnn = GNN(nfeat=512, nhid=args.n_hid, nclass=n_class, dropout=0.5, n_layer=args.n_layer, concat=args.concat).to(device)
			elif args.network == 'gcn':
				gnn = GCN(nfeat=512, nhid=args.n_hid, nclass=n_class, dropout=0.5, n_layer=args.n_layer, concat=args.concat).to(device)
			elif 'gat' in args.network:
				gnn = GAT(nfeat=512, nhid=args.n_hid, nclass=n_class, dropout=0.5, nheads=args.attention_head, alpha=0.2, n_layer=args.n_layer, concat=args.concat, gatType=args.network).to(device)				
			
			optimizer = optim.Adam(list(encoder.parameters()) + list(gnn.parameters()),
									lr=0.01, weight_decay=5e-4)
			
			if args.dataset == 'city':
				criterion = nn.CrossEntropyLoss()
			else:
				criterion = nn.BCELoss()

			if 'gat' in args.network:
				encoder, gnn = train_gat(encoder, gnn, dataloader, optimizer, criterion, nas_dir, device, i, epochs=args.epochs, concat=args.concat, network=args.network)
			else:
				encoder, gnn = train_gnn(encoder, gnn, dataloader, optimizer, criterion, nas_dir, device, i, epochs=args.epochs, concat=args.concat)
			

			
			
			torch.save(encoder.state_dict(), encoder_path)
			torch.save(gnn.state_dict(), gnn_path)

	
if __name__ == '__main__':
	main()