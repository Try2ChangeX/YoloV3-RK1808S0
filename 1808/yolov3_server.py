import cv2
import numpy as np
from rknn.api import RKNN
from rknn_server_class import rknn_server
from timeit import default_timer as timer

model = './yolov3_tiny.rknn'

GRID0 = 13
GRID1 = 26
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
MAX_BOXES = 500
OBJ_THRESH = 0.2
NMS_THRESH = 0.2
obj_thresh = -np.log(1/OBJ_THRESH - 1)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def process(input, mask, anchors):
	anchors = [anchors[i] for i in mask]
	grid_h, grid_w = map(int, input.shape[0:2])

	box_confidence = input[..., 4]

	pos = np.where(box_confidence > obj_thresh)
	input = input[pos]
	box_confidence = sigmoid(input[..., 4])
	box_confidence = np.expand_dims(box_confidence, axis=-1)

	box_class_probs = sigmoid(input[..., 5:])

	box_xy = sigmoid(input[..., :2])
	box_wh = np.exp(input[..., 2:4])
	for idx, val in enumerate(pos[2]):
		box_wh[idx] = box_wh[idx] * anchors[pos[2][idx]]
	pos0 = np.array(pos[0])[:, np.newaxis]
	pos1 = np.array(pos[1])[:, np.newaxis]
	grid = np.concatenate((pos1, pos0), axis=1)
	box_xy += grid
	box_xy /= (grid_w, grid_h)
	box_wh /= (320, 320)
	box_xy -= (box_wh / 2.)
	box = np.concatenate((box_xy, box_wh), axis=-1)

	return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
	"""Filter boxes with object threshold.

	# Arguments
		boxes: ndarray, boxes of objects.
		box_confidences: ndarray, confidences of objects.
		box_class_probs: ndarray, class_probs of objects.

	# Returns
		boxes: ndarray, filtered boxes.
		classes: ndarray, classes for boxes.
		scores: ndarray, scores for boxes.
	"""
	box_scores = box_confidences * box_class_probs
	box_classes = np.argmax(box_scores, axis=-1)
	box_class_scores = np.max(box_scores, axis=-1)
	pos = np.where(box_class_scores >= OBJ_THRESH)

	boxes = boxes[pos]
	classes = box_classes[pos]
	scores = box_class_scores[pos]

	return boxes, classes, scores

def nms_boxes(boxes, scores):
	"""Suppress non-maximal boxes.

	# Arguments
		boxes: ndarray, boxes of objects.
		scores: ndarray, scores of objects.

	# Returns
		keep: ndarray, index of effective boxes.
	"""
	x = boxes[:, 0]
	y = boxes[:, 1]
	w = boxes[:, 2]
	h = boxes[:, 3]

	areas = w * h
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
		h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
		inter = w1 * h1

		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= NMS_THRESH)[0]
		order = order[inds + 1]
	keep = np.array(keep)
	return keep


def yolov3_post_process(input_data):
	# # yolov3
	# masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	# anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
	#			 [59, 119], [116, 90], [156, 198], [373, 326]]
	# yolov3-tiny
	masks = [[3, 4, 5], [0, 1, 2]]
	anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

	boxes, classes, scores = [], [], []
	for input,mask in zip(input_data, masks):
		b, c, s = process(input, mask, anchors)
		b, c, s = filter_boxes(b, c, s)
		boxes.append(b)
		classes.append(c)
		scores.append(s)

	boxes = np.concatenate(boxes)
	classes = np.concatenate(classes)
	scores = np.concatenate(scores)

	# # Scale boxes back to original image shape.
	# width, height = 416, 416 #shape[1], shape[0]
	# image_dims = [width, height, width, height]
	# boxes = boxes * image_dims

	nboxes, nclasses, nscores = [], [], []
	for c in set(classes):
		inds = np.where(classes == c)
		b = boxes[inds]
		c = classes[inds]
		s = scores[inds]

		keep = nms_boxes(b, s)

		nboxes.append(b[keep])
		nclasses.append(c[keep])
		nscores.append(s[keep])

	if not nclasses and not nscores:
		return None, None, None

	boxes = np.concatenate(nboxes)
	classes = np.concatenate(nclasses)
	scores = np.concatenate(nscores)

	return boxes, classes, scores

def post_process(output):
	out_boxes = output[0]
	out_boxes2 = output[1]

	out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
	out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)
	input_data = []
	input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
	input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))

	boxes, classes, scores = yolov3_post_process(input_data)
	print(type(boxes))

	print(type(classes))

	print(type(scores))


	return boxes, classes, scores

if __name__ == '__main__':
	rknn = rknn_server(8002)

	rknn.service(model, post_process)

