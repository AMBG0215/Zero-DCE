import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 

def lowlight(image_path, device, checkpoint='snapshots/Epoch99.pth',
			 input_base=os.path.join('data', 'test_data'), result_base=os.path.join('data', 'result')):

	data_lowlight = Image.open(image_path)

	arr = np.asarray(data_lowlight).astype(np.float32) / 255.0
	# If grayscale, expand to 3 channels
	if arr.ndim == 2:
		arr = np.stack([arr, arr, arr], axis=-1)

	data_lowlight = torch.from_numpy(arr).float()
	data_lowlight = data_lowlight.permute(2, 0, 1)
	data_lowlight = data_lowlight.to(device).unsqueeze(0)

	DCE_net = model.enhance_net_nopool().to(device)
	DCE_net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
	start = time.time()
	_, enhanced_image, _ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)

	# compute result path relative to test base and create directories cross-platform
	rel = os.path.relpath(image_path, start=input_base)
	result_path = os.path.join(result_base, rel)
	os.makedirs(os.path.dirname(result_path), exist_ok=True)

	# move to CPU and save
	torchvision.utils.save_image(enhanced_image.cpu(), result_path)


def collect_image_paths(input_path):
	image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
	if os.path.isfile(input_path):
		return [input_path]

	paths = []
	for root, _, files in os.walk(input_path):
		for file_name in files:
			if file_name.lower().endswith(image_extensions):
				paths.append(os.path.join(root, file_name))
	return paths

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
						help='device to run on (auto selects CUDA if available)')
	parser.add_argument('--input', default=os.path.join('data', 'test_data'),
						help='input image file or directory')
	parser.add_argument('--checkpoint', default='snapshots/Epoch99.pth')
	args = parser.parse_args()

	# decide device and print helpful diagnostic if CUDA requested but unavailable
	if args.device == 'auto':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif args.device == 'cuda':
		if not torch.cuda.is_available():
			print('Requested CUDA but torch reports CUDA unavailable.')
			print('torch.cuda.is_available():', torch.cuda.is_available())
			print('torch.version.cuda:', torch.version.cuda)
			print('torch.__version__:', torch.__version__)
			print('If you installed system CUDA, install a matching PyTorch build with CUDA support.')
			device = torch.device('cpu')
		else:
			device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	with torch.no_grad():
		input_path = args.input
		result_base = os.path.join('data', 'result')
		image_paths = collect_image_paths(input_path)

		if not image_paths:
			print('No images found in:', input_path)
		else:
			for image in image_paths:
				print(image)
				base = os.path.dirname(image) if os.path.isfile(input_path) else input_path
				lowlight(image, device, checkpoint=args.checkpoint, input_base=base,
						 result_base=result_base)



