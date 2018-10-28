import os
import yaml
import argparse
import numpy as np
import tensorflow as tf

from kpcn import KPCN
import data_utils as du

class Denoiser(object):
	"""
	KPCN denoising pipeline.
	"""

	def __init__(self, config):
		buffer_h = config['data']['buffer_h']
		buffer_w = config['data']['buffer_w']
		layers_config = config['layers']
		is_training = config['op'] == 'train'
		learning_rate = config['train_params']['learning_rate']
		summary_dir = config['train_params'].get('summary_dir', 'summaries')

		fp16 = config['train_params'].get('fp16', True)
		tf_dtype = tf.float16 if fp16 else tf.float32

		if type(learning_rate) == str:
			learning_rate = eval(learning_rate)

		self.eps = 0.00316
		self.fp16 = fp16
		self.diff_kpcn = None
		self.spec_kpcn = None
		if config['kpcn']['incl_diff']:
			self.diff_kpcn = KPCN(buffer_h, buffer_w, layers_config, is_training, learning_rate, summary_dir, scope='diffuse', dtype=tf_dtype)
		if config['kpcn']['incl_spec']:
			self.spec_kpcn = KPCN(buffer_h, buffer_w, layers_config, is_training, learning_rate, summary_dir, scope='specular', dtype=tf_dtype)

	def train(self, config):
		log_freq = config['train_params'].get('log_freq', 100)
		save_freq = config['train_params'].get('save_freq', 1000)
		checkpoint_dir = config['train_params']['checkpoint_dir']
		max_steps = config['train_params'].get('max_steps', 1e9)

		if type(max_steps) == str:
			max_steps = int(eval(max_steps))

		# load data
		batched_buffers_diff, batched_buffers_spec, gt_out_diff, gt_out_spec = self.load_data(config)

		tf_config = tf.ConfigProto(
			device_count={'GPU': 1}, allow_soft_placement=True)
		with tf.Session(config=tf_config) as sess:
			# initialization
			sess.run(tf.global_variables_initializer())

			# training loop
			for i in range(max_steps):
				if config['kpcn']['incl_diff']:
					diff_loss = self.diff_kpcn.run_train_step(sess, batched_buffers_diff, gt_out_diff, i)
					if (i + 1) % log_freq == 0:
						print('[step %d] diff loss: %.5f' % (i, diff_loss))
					if (i + 1) % save_freq == 0:
						self.diff_kpcn.save(sess, i, checkpoint_dir=os.path.join(checkpoint_dir, 'diff_kpcn'))

				if config['kpcn']['incl_spec']:
					spec_loss = self.spec_kpcn.run_train_step(sess, batched_buffers_spec, gt_out_spec, i)
					if (i + 1) % log_freq == 0:
						print('[step %d] spec loss: %.5f' % (i, spec_loss))
					if (i + 1) % save_freq == 0:
						self.spec_kpcn.save(sess, i, checkpoint_dir=os.path.join(checkpoint_dir, 'spec_kpcn'))

	def load_data(self, config):
		batch_size = config['train_params'].get('batch_size', 5)

		# the following code will change when I actually have a dataset
		def parse_exr_file(filepath):
			buffers = du.read_exr(filepath, fp16=self.fp16)
			buffers = du.stack_channels(buffers)

			# preprocess
			du.preprocess_diffuse(buffers, self.eps)
			du.preprocess_specular(buffers)
			du.preprocess_depth(buffers)
			grad_y, grad_x = du.compute_buffer_gradients(buffers)

			var_features = np.concatenate([
				buffers['normalVariance'],
				buffers['albedoVariance'],
				buffers['depthVariance']], axis=-1)
			grad_y_features = np.concatenate([
				grad_y['normal'], grad_y['albedo'], grad_y['depth']], axis=-1)
			grad_x_features = np.concatenate([
				grad_x['normal'], grad_x['albedo'], grad_x['depth']], axis=-1)

			buffers_diff = {
				'color': buffers['diffuse'],
				'grad_x': np.concatenate([grad_x['diffuse'], grad_x_features], axis=-1),
				'grad_y': np.concatenate([grad_y['diffuse'], grad_y_features], axis=-1),
				'var_color': buffers['diffuseVariance'],
				'var_features': var_features,
			}
			buffers_spec = {
				'color': buffers['specular'],
				'grad_x': np.concatenate([grad_x['specular'], grad_x_features], axis=-1),
				'grad_y': np.concatenate([grad_y['specular'], grad_y_features], axis=-1),
				'var_color': buffers['specularVariance'],
				'var_features': var_features,
			}

			# add batch dim
			for c, data in buffers_diff.items():
				buffers_diff[c] = np.expand_dims(data, 0)
			for c, data in buffers_spec.items():
				buffers_spec[c] = np.expand_dims(data, 0)

			# [temp] take 65x65 patch
			for c, data in buffers_diff.items():
				buffers_diff[c] = data[:, 300:365, 600:665, :]
			for c, data in buffers_spec.items():
				buffers_spec[c] = data[:, 300:365, 600:665, :]

			return buffers_diff, buffers_spec

		in_path = '/home/owen/data/house-00128spp.exr'
		gt_path = '/home/owen/data/house-08192spp.exr'
		in_batched_buffers_diff, in_batched_buffers_spec = parse_exr_file(in_path)
		gt_batched_buffers_diff, gt_batched_buffers_spec = parse_exr_file(gt_path)

		gt_out_diff = gt_batched_buffers_diff['color']
		gt_out_spec = gt_batched_buffers_spec['color']

		print('[o] Loaded data.')
		return in_batched_buffers_diff, in_batched_buffers_spec, gt_out_diff, gt_out_spec

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config path')
	args = parser.parse_args()

	assert os.path.isfile(args.config)
	config = yaml.load(open(args.config, 'r'))
	denoiser = Denoiser(config)
	getattr(denoiser, config['op'])(config)
