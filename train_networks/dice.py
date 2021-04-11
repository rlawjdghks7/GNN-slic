
def dice_score(pred, target, smooth = 1.): # for 2D image
	intersection = (pred * target).sum(-1).sum(-1)
	union = (pred + target).sum(-1).sum(-1)
	# print(union.shape)

	dice = (2. * intersection + smooth) / (union + smooth)
	# print(dice.mean())
	return dice.mean()

def IoU(pred, target, smooth = 1.): # for 2D image
	intersection = (pred * target).sum(-1).sum(-1)
	union = (pred + target).sum(-1).sum(-1)
	# print(union.shape)

	iou = intersection / union
	# print(dice.mean())
	return iou.mean()