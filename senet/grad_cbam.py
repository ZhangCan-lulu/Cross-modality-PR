import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
import torch.nn as nn




class FeatureExtractor():
	""" Class for extracting activations and
    registering gradients from targetted intermediate layers """
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []

		x=self.model.visible_net(x)
		x=self.model.glo_avgpool(x)
		x=self.model.pyramid_attention(x)
		x = x.view(x.size(0),x.size(1))

		for name, module in self.model._modules.items():

			print('name=',name)
			print('x.size()=',x.size())
			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
		return outputs, x


class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model

		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.glo_avgpool = nn.AdaptiveAvgPool2d((1, 1))
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)

		output = self.model.classifier(output)
		return target_activations, output


class GradCam:
	def __init__(self, model, target_layer_names,use_cuda):
		self.model = model
		self.model.eval()

		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):

		features, output = self.extractor(input)


		if index == None:
			index = np.argmax(output.cpu().data.numpy(),1)

		one_hot = np.zeros((output.size(0), output.size(1)), dtype = np.float32)
		for i in range(output.size(0)):
			one_hot[i][index[i]]=1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()

		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[-1, :]

		weights = np.mean(grads_val, axis = (2, 3))[-1, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (144, 288))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam



class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.backbone._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.backbone._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output


def show_cam_on_image(img, mask,mode='visible'):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	img_tmp=np.float32(img).transpose((1,2,0))[:,:,::-1]
	cam = heatmap + img_tmp
	# cam = cam / np.max(cam)

	cv2.imwrite("img" + mode + ".jpg", np.uint8(img_tmp))
	cv2.imwrite("heatmap" + mode + ".jpg", np.uint8(heatmap))
	cv2.imwrite("cam"+mode+".jpg", np.uint8(255 * cam))




def returncCAM(feature,class_idx):
	size_upsample =(288,144)
	bz, nc, h, w = feature.shape
	for idx in class_idx:
		cam = feature[idx].reshape((nc, h * w))
		cam = cam.reshape(h, w)
		cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))

	return output_cam