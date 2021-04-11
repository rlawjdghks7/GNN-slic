import torch
import numpy as np
import matplotlib.pyplot as plt

import sys

def gen_graph_label(label, slic_arr):
	label = torch.squeeze(label)

	graph_label = []
	for superpixel_val in np.unique(slic_arr):
		mask = slic_arr == superpixel_val
		temp_map = label[mask]
		node = 1.0 if temp_map.mean() > 0.1 else 0.0
		graph_label.append(node)
	graph_label = np.array(graph_label)
	graph_label = torch.from_numpy(graph_label).unsqueeze(1)
    
	return graph_label