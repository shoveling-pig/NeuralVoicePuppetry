import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import glob
from tqdm import tqdm

from BaselModel.basel_model import MorphableModel


def plot_mesh(vertices, triangles, subplot = [1,1,1], title = 'mesh', el = 90, az = -90, lwdt=.1, dist = 6, color = "grey"):
	'''
	plot the mesh 
	Args:
		vertices: [nver, 3]
		triangles: [ntri, 3]
	'''
	ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection = '3d')
	ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles = triangles, lw = lwdt, color = color, alpha = 1)
	ax.axis("off")
	ax.view_init(elev = el, azim = az)
	ax.dist = dist
   

def generateGIF(scale, expression_path):
	mm = MorphableModel()

	output_path1 = '/data3/shovelingpig/STV/NeuralVoicePuppetry/output/average.obj'
	mm.save_model_to_obj_file(output_path1)


	with open(expression_path, 'r') as f:
		expressions = []
		for i, line in enumerate(f):
			expression = line.strip().split(' ')
			expression = [scale * float(x) for x in expression if len(x.strip()) != 0]
			expressions.append(expression)
		expressions = torch.Tensor(expressions).cuda()

	mm.morph(expressions)
	output_path2 = '/data3/shovelingpig/STV/NeuralVoicePuppetry/output/expressions.obj'
	mm.save_model_to_obj_file(output_path2)

	for i, vertices in enumerate(tqdm(mm.vertices[:30])):
		plot_mesh(vertices.cpu(), mm.faces[0].cpu())
		plt.savefig('/data3/shovelingpig/STV/NeuralVoicePuppetry/output/image/test{}.png'.format(i))
		plt.clf()

	frames = []
	imgs = glob.glob("/data3/shovelingpig/STV/NeuralVoicePuppetry/output/image/*.png")
	for i in imgs:
		new_frame = Image.open(i)
		frames.append(new_frame)

	frames[0].save('/data3/shovelingpig/STV/NeuralVoicePuppetry/output/trump_x{}.gif'.format(scale), format='GIF',
				append_images=frames[1:],
				save_all=True,
				duration=300, loop=0)


if __name__ == '__main__':
	# expression_path = '/data3/shovelingpig/STV/NeuralVoicePuppetry/Audio2ExpressionNet/Inference/datasets/TRANSFERS/audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS1-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead/greta_1--Trump/expression.txt'
	expression_path = '/data3/shovelingpig/STV/NeuralVoicePuppetry/Audio2ExpressionNet/Inference/datasets/TARGETS/Trump/expression.txt'
	for scale in [1000, 1200, 1400, 1600, 1800, 2000]:
		print('generate trump_original_x{}.gif...'.format(scale))
		generateGIF(scale, expression_path)
		