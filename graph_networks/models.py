import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_networks.layers import *

class GNN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout, n_layer, concat=False):
		super(GNN, self).__init__()
		self.n_layer = n_layer
		self.concat = concat
		self.nclass = nclass

		if n_layer == 2:
			self.gc1 = GraphLinear(nfeat, nhid)
			self.gc2 = GraphLinear(nhid, nclass)
		elif n_layer == 4:
			self.gc1 = GraphLinear(nfeat, 256)
			self.gc2 = GraphLinear(256, 128)
			self.gc3 = GraphLinear(128, 64)
			
			if concat == True:
				self.gc_down1 = GraphLinear(2*256, 256)
				self.gc_down2 = GraphLinear(2*128, 128)
				self.gc_down3 = GraphLinear(2*64, 64)
			self.gc4 = GraphLinear(64, nclass)

		self.dropout = dropout

	def forward(self, x, adj, ft_to_graph_list=None):
		gft_list = []

		if self.n_layer == 2:
			x = F.relu(self.gc1(x))
			gft_list.append(x)
			x = self.gc2(x)
		elif self.n_layer == 4:
			if self.concat == True:
				# original x is n_superpix x 512
				x = F.relu(self.gc1(x)) # n_superpix x 256
				x = torch.cat((x, ft_to_graph_list[-2]), dim=1) # n_superpix x 512
				x = F.relu(self.gc_down1(x)) # n_superpix x 256
				gft_list.append(x)

				x = F.relu(self.gc2(x)) # n_superpix x 128
				x = torch.cat((x, ft_to_graph_list[-3]), dim=1) # n_superpix x 256
				x = F.relu(self.gc_down2(x)) # n_superpix x 128
				gft_list.append(x)

				x = F.relu(self.gc3(x)) # n_superpix x 64
				x = torch.cat((x, ft_to_graph_list[-4]), dim=1) # n_superpix x 128
				x = F.relu(self.gc_down3(x)) # n_superpix x 64
				gft_list.append(x)

				x = self.gc4(x)  # n_superpix x 1
				# gft_list.append(x)
			else:
				x = F.relu(self.gc1(x))
				gft_list.append(x)
				x = F.relu(self.gc2(x))
				gft_list.append(x)
				x = F.relu(self.gc3(x))
				gft_list.append(x)
				x = self.gc4(x)
				
		if self.nclass == 1:
			x = torch.sigmoid(x)
		else:
			# x = torch.softmax(x, dim=1)
			x = F.log_softmax(x, dim=1)
		# return F.log_softmax(x, dim=1)
		return x, gft_list[:]
		# return x

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout, n_layer, concat=False):
		super(GCN, self).__init__()
		self.n_layer = n_layer
		self.concat = concat

		if n_layer == 2:
			self.gc1 = GraphConvolution(nfeat, nhid)
			self.gc2 = GraphConvolution(nhid, nclass)
		elif n_layer == 4:
			self.gc1 = GraphConvolution(nfeat, 256)
			self.gc2 = GraphConvolution(256, 128)
			self.gc3 = GraphConvolution(128, 64)
			
			if concat == True:
				self.gc_down1 = GraphConvolution(2*256, 256)
				self.gc_down2 = GraphConvolution(2*128, 128)
				self.gc_down3 = GraphConvolution(2*64, 64)
			self.gc4 = GraphConvolution(64, nclass)
		self.dropout = dropout

	def forward(self, x, adj, ft_to_graph_list=None):
		gft_list = []
		if self.n_layer == 2:
			x = F.relu(self.gc1(x, adj))
			gft_list.append(x)
			# x = F.dropout(x, self.dropout, training=self.training)
			x = torch.sigmoid(self.gc2(x, adj))
		elif self.n_layer == 4:
			if self.concat == True:
				# print('in gcn concat!')
				x = F.relu(self.gc1(x, adj))
				x = torch.cat((x, ft_to_graph_list[-2]), dim=1) # n_superpix x 512
				x = F.relu(self.gc_down1(x, adj)) # n_superpix x 256
				gft_list.append(x)

				x = F.relu(self.gc2(x, adj))
				x = torch.cat((x, ft_to_graph_list[-3]), dim=1) # n_superpix x 512
				x = F.relu(self.gc_down2(x, adj)) # n_superpix x 256
				gft_list.append(x)

				x = F.relu(self.gc3(x, adj))
				x = torch.cat((x, ft_to_graph_list[-4]), dim=1) # n_superpix x 512
				x = F.relu(self.gc_down3(x, adj)) # n_superpix x 256
				gft_list.append(x)

				x = torch.sigmoid(self.gc4(x, adj))

			else:
				# print('not gcn concat!')
				x = F.relu(self.gc1(x, adj))
				gft_list.append(x)
				x = F.relu(self.gc2(x, adj))
				gft_list.append(x)
				x = F.relu(self.gc3(x, adj))
				gft_list.append(x)
				x = torch.sigmoid(self.gc4(x, adj))
			
		return x, gft_list[:]

class GAT(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_layer, concat=False, gatType='gat'):
		"""Dense version of GAT."""
		super(GAT, self).__init__()
		self.dropout = dropout
		self.n_layer = n_layer
		self.concat = concat
		self.gatType = gatType

		if n_layer == 2:
			self.attentions = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)
			self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False, gatType=self.gatType)
		
		elif n_layer == 4:
			self.attentions = GraphAttentionLayer(nfeat, 256, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)
			self.attentions2 = GraphAttentionLayer(256, 128, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)
			self.attentions3 = GraphAttentionLayer(128, nhid, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)

			if concat == True:
				self.att_down1 = GraphAttentionLayer(256 + 256, 256, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)
				self.att_down2 = GraphAttentionLayer(128 + 128, 128, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)
				self.att_down3 = GraphAttentionLayer(64 + 64, 16, dropout=dropout, alpha=alpha, concat=True, gatType=self.gatType)

			self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False, gatType=self.gatType)

	def forward(self, x, adj, ft_to_graph_list=None):
		gft_list = []
		att_list = []
		if self.n_layer == 2:
			x, attention = self.attentions(x, adj)
			gft_list.append(x)
			att_list.append(attention)

		elif self.n_layer == 4:
			if self.concat == True:
				x, attention = self.attentions(x, adj)
				x = torch.cat((x, ft_to_graph_list[-2]), dim=1)
				x, attention = self.att_down1(x, adj)
				att_list.append(attention)
				gft_list.append(x)

				x, attention = self.attentions2(x, adj)
				x = torch.cat((x, ft_to_graph_list[-3]), dim=1)
				x, attention = self.att_down2(x, adj)
				att_list.append(attention)
				gft_list.append(x)

				x, attention = self.attentions3(x, adj)
				x = torch.cat((x, ft_to_graph_list[-4]), dim=1)
				x, attention = self.att_down3(x, adj)
				att_list.append(attention)
				gft_list.append(x)
			else:
				x, attention = self.attentions(x, adj)
				gft_list.append(x)
				att_list.append(attention)
				x, attention = self.attentions2(x, adj)
				gft_list.append(x)
				att_list.append(attention)
				x, attention = self.attentions3(x, adj)
				gft_list.append(x)
				att_list.append(attention)
				
		x, attention = self.out_att(x, adj)
		x = torch.sigmoid(x)
		return x, gft_list, att_list, attention

class GraphUnet(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout, alpha, n_layer, concat=False, gatType='newgat3'):
		"""Dense version of GAT."""
		super(GraphUnet, self).__init__()
		self.n_layer = n_layer
		# self.ks = ks
		self.start_gat = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, gatType=gatType)
		self.bottom_gat = GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, gatType=gatType)

		self.down_gats = nn.ModuleList()
		self.up_gats = nn.ModuleList()
		self.pools = nn.ModuleList()
		self.unpools = nn.ModuleList()
		for i in range(n_layer):
			self.down_gats.append(GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, gatType=gatType))
			self.up_gats.append(GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, gatType=gatType))
			self.pools.append(Pool(0.5, nhid))
			self.unpools.append(Unpool())
		
		self.out_linear = GraphLinear(nhid, nclass)

	def forward(self, x, adj, ft_to_graph_list=None):
		gft_down_list = []
		gft_up_list = []
		score_list = []
		recon_adj_list = []

		att_list = []

		adj_ms = []
		indices_list = []
		down_outs = []
		xs = []
		
		# print(adj.size())
		# sys.exit()
		x, att = self.start_gat(x, adj)
		# gft_down_list.append(x)
		origin_size = (adj.shape[0], x.shape[1])
		org_x = x
		for i in range(self.n_layer):
			x, _ = self.down_gats[i](x, adj)
			adj_ms.append(adj)
			down_outs.append(x)

			adj, x, idx, scores = self.pools[i](adj, x)
			indices_list.append(idx)
			full_x = x.new_zeros([origin_size[0], origin_size[1]])
			org_idx = recursive_index(indices_list)
			full_x[org_idx] = x
			gft_down_list.append(full_x)

			# a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
			# b = np.array([1,3])
			# print(a)
			# print(a[b,:])
			# print(a[:,b])
			# print(a[b,b])
			# print(a[b][:,b])

			recon_adj = torch.zeros_like(adj_ms[0])
			# print(recon_adj.size())
			# print(adj.size())
			# print(org_idx.size())
			# print(recon_adj[org_idx,org_idx].size())

			# recon_adj[org_idx][:,org_idx] = 1
			recon_adj = copy_adj(recon_adj, adj, org_idx)
			# print(recon_adj[org_idx][:,org_idx])
			# print(recon_adj[org_idx][:,org_idx].size())
			# print(torch.where(recon_adj>0))
			# print(adj)
			# # print(recon_adj.size())
			# sys.exit()

			recon_adj_list.append(recon_adj)

			if i >= 1:
				score_x = scores.new_zeros([origin_size[0]])
				org_idx = recursive_index(indices_list[:-1])
				score_x[org_idx] = scores
				score_list.append(score_x.unsqueeze(1))
			else:
				score_list.append(scores.unsqueeze(1))


		x, _ = self.bottom_gat(x, adj)

		for i in range(self.n_layer):
			up_idx = self.n_layer - i - 1
			adj, idx = adj_ms[up_idx], indices_list[up_idx]
			adj, x = self.unpools[i](adj, x, down_outs[up_idx], idx)
			# unpool did not use previous feature_map?
			x, att = self.up_gats[i](x, adj)
			if i < self.n_layer-1:
				new_x = x.new_zeros([origin_size[0], origin_size[1]])
				pre_idx = recursive_index(indices_list[:up_idx])
				new_x[pre_idx] = x
				gft_up_list.append(new_x)
			else:
				gft_up_list.append(x)
			# x = x.add(down_outs[up_idx])
			# xs.append(x)
		# x = x.add(org_x)
		# gft_list.append(x)
		# xs.append(x)
		# classify
		x = self.out_linear(x)
		x = torch.sigmoid(x)
		return x, gft_down_list, gft_up_list, att, score_list, recon_adj_list

def copy_adj(recon, adj, idx):
	# x, y = recon.size()
	# print(f'recon size: {recon.size()}')
	# print(f'adj size: {adj.size()}')
	# print(f'idx size: {idx.size()}')
	for i in range(len(idx)):
		for j in range(len(idx)):
			p_x, p_y = idx[i], idx[j]
			# try:
			recon[p_x, p_y] = adj[i, j]
			# except:
			# 	print(p_x, p_y)
			# 	print(i, j)
			# 	sys.exit()
	return recon

def recursive_index(indices_list):
	if len(indices_list) == 1:
		return indices_list[-1]
	else:
		return indices_list[0][(recursive_index(indices_list[1:]))]