import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

import sys

class GraphAttentionLayer(nn.Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""
	def __init__(self, in_features, out_features, dropout, alpha, concat=True, bias=False, gatType='gat'):
		super(GraphAttentionLayer, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		self.gatType = gatType

		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		nn.init.xavier_uniform_(self.weight.data, gain=1.414)
		self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):

		if 'gat2' in self.gatType:
			# print('cur gat2!')
			e = self._prepare_attentional_mechanism_input(h)

			if not 'newgat' in self.gatType:
				e = self.leakyrelu(torch.matmul(e, self.a).squeeze(2))
			zero_vec = -9e15*torch.ones_like(e)
			attention = torch.where(adj > 0, e, zero_vec)
			attention = F.softmax(attention, dim=1)

			h_prime = torch.matmul(attention, h)
			h_prime = torch.mm(h_prime, self.weight)

		else:
			# print('cur gat!')
			Wh = torch.mm(h, self.weight) # h.shape: (N, in_features), Wh.shape: (N, out_features)
			
			e = self._prepare_attentional_mechanism_input(Wh)
			
			if not 'newgat' in self.gatType:
				e = self.leakyrelu(torch.matmul(e, self.a).squeeze(2))

			if 'gat3' in self.gatType:
				zero_vec = torch.zeros_like(e)
				e_adj = torch.where(adj > 0, e, zero_vec)
				# if len(e_adj[0]) > 32:
				# 	print('*'*20)
				# 	print(e_adj[31])
				sum_result = torch.sum(e_adj, dim=1).unsqueeze(0)
				attention = e_adj / sum_result.transpose(1, 0)
				# if len(e_adj[0]) > 32:
				# 	print(attention[31])
			else:
				zero_vec = -9e15*torch.ones_like(e)
				attention = torch.where(adj > 0, e, zero_vec)
				attention = F.softmax(attention, dim=1)

			# a = np.arange(0, 16)
			# a = a.reshape(4, 4)
			# a = torch.FloatTensor(a)
			# print(a)
			# print(a[0])
			# # sum_result = torch.sum(a, dim=1)
			# sum_result = torch.sum(a, dim=1).unsqueeze(0)
			# print(sum_result)
			# print(a/sum_result.transpose(1, 0))
			# sys.exit()
			
			h_prime = torch.matmul(attention, Wh)

		if self.concat:
			return F.elu(h_prime), attention
		else:
			return h_prime, attention

	def _prepare_attentional_mechanism_input(self, Wh):
		N = Wh.size()[0] # number of nodes

		# Below, two matrices are created that contain embeddings in their rows in different orders.
		# (e stands for embedding)
		# These are the rows of the first matrix (Wh_repeated_in_chunks): 
		# e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
		# '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
		# 
		# These are the rows of the second matrix (Wh_repeated_alternating): 
		# e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
		# '----------------------------------------------------' -> N times
		# 
		
		Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
		
		Wh_repeated_alternating = Wh.repeat(N, 1)

		# Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

		# The all_combination_matrix, created below, will look like this (|| denotes concatenation):
		# e1 || e1
		# e1 || e2
		# e1 || e3
		# ...
		# e1 || eN
		# e2 || e1
		# e2 || e2
		# e2 || e3
		# ...
		# e2 || eN
		# ...
		# eN || e1
		# eN || e2
		# eN || e3
		# ...
		# eN || eN

		if 'newgat' in self.gatType:
			e = torch.exp((-1) * torch.norm(Wh_repeated_alternating-Wh_repeated_in_chunks, dim=1))
			# diff = torch.mean(torch.pow(Wh_repeated_alternating - Wh_repeated_in_chunks, 2), dim=1)
			# e = torch.exp((-1) * diff)

			result = e.view(N, N)

		else:
			all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
			result = all_combinations_matrix.view(N, N, 2 * self.out_features)

		return result

	# def __repr__(self):
	# 	return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj.t(), support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'

class GraphLinear(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		output = torch.mm(input, self.weight)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'

class Pool(nn.Module):
	def __init__(self, k, in_dim, p=0.9):
		super(Pool, self).__init__()
		self.k = k
		self.sigmoid = nn.Sigmoid()
		self.proj = nn.Linear(in_dim, 1)
		# self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

	def forward(self, g, h): # g : adj, h : graph feature
		weights = self.proj(h).squeeze()
		scores = self.sigmoid(weights)
		# scores = torch.mean(h, dim=1)
		g, new_h, idx = top_k_graph(scores, g, h, self.k)

		return g, new_h, idx, scores

class Unpool(nn.Module):
	def __init__(self, *args):
		super(Unpool, self).__init__()

	def forward(self, g, h, pred_h, idx):
		new_h = h.new_zeros([g.shape[0], h.shape[1]])
		# new_h = pred_h.clone().detach()
		new_h = new_h.add(pred_h)

		new_h[idx] = h
		return g, new_h

def top_k_graph(scores, g, h, k):
	num_nodes = g.shape[0]
	values, idx = torch.topk(scores, max(2, int(k*num_nodes)), sorted=False)
	new_h = h[idx, :]
	# values = torch.unsqueeze(values, -1)
	# new_h = torch.mul(new_h, values)
	un_g = g.bool().float()
	# un_g = torch.matmul(un_g, un_g).bool().float()
	un_g = un_g[idx, :]
	un_g = un_g[:, idx]
	g = norm_g(un_g)
	return g, new_h, idx

def norm_g(g):
	degrees = torch.sum(g, 1)
	g = g / degrees
	return g