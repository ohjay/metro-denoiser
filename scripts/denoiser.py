import os
import sys
import glob
import yaml
import time
import shutil
import random
import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imsave

from kpcn import KPCN
import data_utils as du

class Denoiser(object):
    """
    KPCN denoising pipeline.
    """

    def __init__(self, config):
        self.eps = 0.00316
        self.diff_kpcn = None
        self.spec_kpcn = None
        self.fp16 = config['train_params'].get('fp16', True)
        self.tf_dtype = tf.float16 if self.fp16 else tf.float32

    @staticmethod
    def _training_loop(sess, kpcn, train_init_op, val_init_op, identifier, log_freq,
                       save_freq, viz_freq, max_epochs, checkpoint_dir, block_on_viz,
                       restore_path='', reset_lr=True):
        # initialize/restore
        sess.run(tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()))
        if os.path.isfile(restore_path + '.index'):
            kpcn.restore(sess, restore_path)
        if reset_lr:
            sess.run(kpcn.global_step.assign(0))  # reset learning rate

        start_time = time.time()

        i = 0
        min_avg_train_loss = float('inf')
        min_avg_val_loss   = float('inf')
        try:
            for epoch in range(max_epochs):
                # Training
                sess.run(train_init_op)
                total_loss, count = 0.0, 0
                while True:
                    try:
                        loss = kpcn.run_train_step(sess, i)
                        total_loss += loss
                        count += 1
                        if (i + 1) % log_freq == 0:
                            avg_loss = total_loss / count
                            if avg_loss < min_avg_train_loss:
                                min_avg_train_loss = avg_loss
                            print('[step %07d] %s loss: %.5f' % (i, identifier, avg_loss))
                            total_loss, count = 0.0, 0
                        if (i + 1) % save_freq == 0:
                            kpcn.save(sess, i, checkpoint_dir=checkpoint_dir)
                            print('[o] Saved model.')
                    except tf.errors.OutOfRangeError:
                        break
                    i += 1
                print('[o][%s] Epoch %d training complete. Running validation...' % (identifier, epoch + 1,))

                # Validation
                sess.run(val_init_op)
                total_loss, count = 0.0, 0
                while True:
                    try:
                        loss, _in, _out, _gt = kpcn.run_validation(sess)
                        if count == 0 and viz_freq > 0 and (i + 1) % viz_freq == 0:
                            du.show_multiple(_in, _out, _gt, block_on_viz=block_on_viz)
                        total_loss += loss
                        count += 1
                    except tf.errors.OutOfRangeError:
                        break
                avg_loss = total_loss / count
                if avg_loss < min_avg_val_loss:
                    min_avg_val_loss = avg_loss
                loss_summary = sess.run(kpcn.loss_summary, {kpcn.loss: avg_loss})
                kpcn.validation_writer.add_summary(loss_summary, i)
                time_elapsed = time.time() - start_time
                print('[o][%s] Validation loss: %.5f' % (identifier, avg_loss))
                print('[o][%s] Epoch %d complete. Time elapsed: %s.' \
                    % (identifier, epoch + 1, du.format_seconds(time_elapsed)))
        except KeyboardInterrupt:
            print('')
            print('[KeyboardInterrupt] summary')
            print('---------------------------')
            print('[o] min avg train loss: %.5f' % min_avg_train_loss)
            print('[o] min_avg_val_loss:   %.5f' % min_avg_val_loss)
            print('[o] total steps:        %d' % i)
            print('---------------------------')
            print('exiting...')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def train(self, config):
        log_freq       = config['train_params'].get('log_freq', 100)
        save_freq      = config['train_params'].get('save_freq', 1000)
        viz_freq       = config['train_params'].get('viz_freq', -1)
        checkpoint_dir = config['train_params']['checkpoint_dir']
        max_epochs     = config['train_params'].get('max_epochs', 1e9)
        batch_size     = config['train_params'].get('batch_size', 5)
        patch_size     = config['data'].get('patch_size', 65)
        layers_config  = config['layers']
        is_training    = config['op'] == 'train'
        learning_rate  = config['train_params']['learning_rate']
        summary_dir    = config['train_params'].get('summary_dir', 'summaries')
        block_on_viz   = config['train_params'].get('block_on_viz', False)
        reset_lr       = config['train_params'].get('reset_lr', True)

        if type(learning_rate) == str:
            learning_rate = eval(learning_rate)
        if type(max_epochs) == str:
            max_epochs = int(eval(max_epochs))

        # load data
        tf_buffers, init_ops = self.load_data(config)

        tf_config = tf.ConfigProto(
            device_count={'GPU': 1}, allow_soft_placement=True)
        with tf.Session(config=tf_config) as sess:
            # training loops
            if config['kpcn']['diff']['train_include']:
                diff_checkpoint_dir = os.path.join(checkpoint_dir, 'diff')
                diff_restore_path   = config['kpcn']['diff'].get('restore_path', '')

                self.diff_kpcn = KPCN(
                    tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='diffuse')
                self._training_loop(sess, self.diff_kpcn, init_ops['diff_train'], init_ops['diff_val'], 'diff',
                    log_freq, save_freq, viz_freq, max_epochs, diff_checkpoint_dir, block_on_viz, diff_restore_path, reset_lr)

            if config['kpcn']['spec']['train_include']:
                spec_checkpoint_dir = os.path.join(checkpoint_dir, 'spec')
                spec_restore_path   = config['kpcn']['spec'].get('restore_path', '')

                self.spec_kpcn = KPCN(
                    tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='specular')
                self._training_loop(sess, self.spec_kpcn, init_ops['spec_train'], init_ops['spec_val'], 'spec',
                    log_freq, save_freq, viz_freq, max_epochs, spec_checkpoint_dir, block_on_viz, spec_restore_path, reset_lr)

    def load_data(self, config, shuffle=True):
        batch_size   = config['train_params'].get('batch_size', 5)
        patch_size   = config['data'].get('patch_size', 65)
        tfrecord_dir = config['data']['tfrecord_dir']
        scenes       = config['data']['scenes']
        clip_ims     = config['data']['clip_ims']

        train_filenames = [
            os.path.join(tfrecord_dir, scene, 'train', 'data.tfrecords') for scene in scenes]
        val_filenames = [
            os.path.join(tfrecord_dir, scene, 'validation', 'data.tfrecords') for scene in scenes]

        with tf.device('/cpu:0'):
            train_dataset = tf.data.TFRecordDataset(train_filenames)
            val_dataset   = tf.data.TFRecordDataset(val_filenames)

            pstep = config['data'].get('parse_step', 1)
            patches_per_im = config['data'].get('patches_per_im', 400)
            train_dataset_size_estimate = (int(200 / pstep) * patches_per_im) * len(train_filenames)
            val_dataset_size_estimate   = (int(200 / pstep) * patches_per_im) * len(val_filenames)
            size_lim = config['data'].get('shuffle_buffer_size_limit', 80000)

            decode_diff = du.make_decode(True, self.tf_dtype, patch_size, patch_size, self.eps, clip_ims)
            decode_spec = du.make_decode(False, self.tf_dtype, patch_size, patch_size, self.eps, clip_ims)
            datasets = {
                'diff_train': train_dataset.map(decode_diff, num_parallel_calls=4),
                'spec_train': train_dataset.map(decode_spec, num_parallel_calls=4),
                'diff_val': val_dataset.map(decode_diff, num_parallel_calls=4),
                'spec_val': val_dataset.map(decode_spec, num_parallel_calls=4),
            }
            for comp in datasets:
                if comp.endswith('train'):
                    dataset_size = min(train_dataset_size_estimate, size_lim)
                else:
                    dataset_size = min(val_dataset_size_estimate, size_lim)
                datasets[comp] = datasets[comp].cache()
                if shuffle:
                    datasets[comp] = datasets[comp].shuffle(dataset_size)
                datasets[comp] = datasets[comp].batch(batch_size)
                datasets[comp] = datasets[comp].prefetch(1)

            # shared iterator
            iterator = tf.data.Iterator.from_structure(
                datasets['diff_train'].output_types, datasets['diff_train'].output_shapes)
            color, normal, albedo, depth, var_color, var_features, gt_out = iterator.get_next()

            # initializers
            diff_train_init_op = iterator.make_initializer(datasets['diff_train'])
            spec_train_init_op = iterator.make_initializer(datasets['spec_train'])
            diff_val_init_op   = iterator.make_initializer(datasets['diff_val'])
            spec_val_init_op   = iterator.make_initializer(datasets['spec_val'])

            # compute gradients
            grad_y_color, grad_x_color = tf.image.image_gradients(color)
            grad_y_normal, grad_x_normal = tf.image.image_gradients(normal)
            grad_y_albedo, grad_x_albedo = tf.image.image_gradients(albedo)
            grad_y_depth, grad_x_depth = tf.image.image_gradients(depth)

            grad_y = tf.concat([grad_y_color, grad_y_normal, grad_y_albedo, grad_y_depth], -1)
            grad_x = tf.concat([grad_x_color, grad_x_normal, grad_x_albedo, grad_x_depth], -1)

        tf_buffers = {
            'color': color,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'var_color': var_color,
            'var_features': var_features,
            'gt_out': gt_out,
        }
        init_ops = {
            'diff_train': diff_train_init_op,
            'spec_train': spec_train_init_op,
            'diff_val': diff_val_init_op,
            'spec_val': spec_val_init_op,
        }
        return tf_buffers, init_ops

    def sample_data(self, config):
        """Write data to TFRecord files for later use."""
        data_dir             = config['data']['data_dir']
        scenes               = config['data']['scenes']
        splits               = config['data']['splits']
        in_filename          = '*-%sspp.exr' % str(config['data']['in_spp']).zfill(5)
        gt_filename          = '*-%sspp.exr' % str(config['data']['gt_spp']).zfill(5)
        tfrecord_dir         = config['data']['tfrecord_dir']
        pstart               = config['data'].get('parse_start', 0)
        pstep                = config['data'].get('parse_step', 1)
        pshuffle             = config['data'].get('parse_shuffle', False)
        patch_size           = config['data'].get('patch_size', 65)
        patches_per_im       = config['data'].get('patches_per_im', 400)
        save_debug_ims       = config['data'].get('save_debug_ims', False)
        save_debug_ims_every = config['data'].get('save_debug_ims_every', 1)
        color_var_weight     = config['data'].get('color_var_weight', 1.0)
        normal_var_weight    = config['data'].get('normal_var_weight', 1.0)

        for scene in scenes:
            input_exr_files = sorted(glob.glob(os.path.join(data_dir, scene, in_filename)))
            gt_exr_files    = sorted(glob.glob(os.path.join(data_dir, scene, gt_filename)))
            # can skip files if on a time/memory budget
            input_exr_files = [f for i, f in enumerate(input_exr_files[pstart:]) if i % pstep == 0]
            gt_exr_files    = [f for i, f in enumerate(gt_exr_files[pstart:])    if i % pstep == 0]
            print('[o] Using %d permutations for scene %s.' % (len(input_exr_files), scene))

            # export dataset generation parameters
            tfrecord_scene_dir = os.path.join(tfrecord_dir, scene)
            if not os.path.exists(tfrecord_scene_dir):
                os.makedirs(tfrecord_scene_dir)
            shutil.copyfile(args.config, os.path.join(tfrecord_scene_dir, 'datagen.yaml'))

            end = 0
            for split in ['train', 'validation', 'test']:  # train gets the first 0.X, val gets the next 0.X, ...
                start = end  # [
                end = start + int(round(splits[split] * len(input_exr_files)))  # )
                if split == 'test' and sum(splits.values()) == 1.0:
                    end = len(input_exr_files)
                print('[o] %s split: %d/%d permutations.' % (split, end - start, len(input_exr_files)))

                _in_files = input_exr_files[start:end]
                _gt_files = gt_exr_files[start:end]
                if pshuffle:
                    _in_gt_paired = list(zip(_in_files, _gt_files))
                    random.shuffle(_in_gt_paired)
                    _in_files, gt_files = zip(*_in_gt_paired)

                tfrecord_scene_split_dir = os.path.join(tfrecord_scene_dir, split)
                if not os.path.exists(tfrecord_scene_split_dir):
                    os.makedirs(tfrecord_scene_split_dir)
                tfrecord_filepath = os.path.join(tfrecord_scene_split_dir, 'data.tfrecords')
                debug_dir = tfrecord_scene_split_dir if save_debug_ims else ''
                du.write_tfrecords(
                    tfrecord_filepath, _in_files, _gt_files, patches_per_im, patch_size, self.fp16,
                    shuffle=pshuffle, debug_dir=debug_dir, save_debug_ims_every=save_debug_ims_every,
                    color_var_weight=color_var_weight, normal_var_weight=normal_var_weight)

    def visualize_data(self, config):
        """Visualize sampled data stored in the TFRecord files."""
        tf_buffers, init_ops = self.load_data(config, shuffle=False)
        tf_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
        with tf.Session(config=tf_config) as sess:
            group = ''
            while group.lower() not in {'diff_train', 'spec_train', 'diff_val', 'spec_val'}:
                group = raw_input('Which data to visualize? {diff,spec}_{train,val} ')
            sess.run(init_ops[group])

            itr_remaining = 1
            while itr_remaining > 0:
                _in, _gt = sess.run([tf_buffers['color'], tf_buffers['gt_out']])
                du.show_multiple(_in, _gt, block_on_viz=True)
                itr_remaining -= 1
                if itr_remaining == 0:
                    try:
                        itr_remaining = int(raw_input('Continue for X iterations: ').strip())
                    except ValueError:
                        print('[o] invalid response (must be integer), exiting...')
                        break

    def denoise(self, config):
        """Run full pipeline to denoise an image."""
        patch_size    = config['data'].get('patch_size', 65)
        layers_config = config['layers']
        is_training   = config['op'] == 'train'
        learning_rate = config['train_params']['learning_rate']
        summary_dir   = config['train_params'].get('summary_dir', 'summaries')

        im_path = config['evaluate']['im_path']
        if im_path.endswith('exr'):
            input_buffers = read_exr(input_filepath, fp16=self.fp16)
            input_buffers = stack_channels(input_buffers)
        else:
            dot_idx      = im_path.rfind('.')
            filetype_str = ' ' + im_path[dot_idx:] if dot_idx != -1 else ''
            raise TypeError('[-] Filetype%s not supported yet.' % filetype_str)

        h, w = input_buffers['diffuse'].shape[:2]

        # clip
        if config['data']['clip_ims']:
            input_buffers['diffuse'] = np.clip(input_buffers['diffuse'], 0.0, 1.0)
            input_buffers['specular'] = np.clip(input_buffers['specular'], 0.0, 1.0)

        # preprocess
        du.preprocess_diffuse(input_buffers, self.eps)
        du.preprocess_specular(input_buffers)
        du.preprocess_depth(input_buffers)

        # make network inputs
        diff_in, spec_in = du.make_network_inputs(input_buffers)

        # define networks
        tf_placeholders = {
            'color':        tf.placeholder(self.tf_dtype, shape=(None, buffer_h, buffer_w, 3))
            'grad_x':       tf.placeholder(self.tf_dtype, shape=(None, buffer_h, buffer_w, 10))
            'grad_y':       tf.placeholder(self.tf_dtype, shape=(None, buffer_h, buffer_w, 10))
            'var_color':    tf.placeholder(self.tf_dtype, shape=(None, buffer_h, buffer_w, 1))
            'var_features': tf.placeholder(self.tf_dtype, shape=(None, buffer_h, buffer_w, 3))
        }
        self.diff_kpcn = KPCN(
            tf_placeholders, patch_size, patch_size, layers_config,
            is_training, learning_rate, summary_dir, scope='diffuse')
        self.spec_kpcn = KPCN(
            tf_placeholders, patch_size, patch_size, layers_config,
            is_training, learning_rate, summary_dir, scope='specular')

        ks = self.diff_kpcn.kernel_size

        # split image into l/r sub-images (with overlap)
        # because my GPU doesn't have enough memory to deal with the whole image at once
        l_diff_in = {c: data[:, :, :w//2+ks, :] for c, data in diff_in.items()}
        r_diff_in = {c: data[:, :, w//2-ks:, :] for c, data in diff_in.items()}
        l_spec_in = {c: data[:, :, :w//2+ks, :] for c, data in spec_in.items()}
        r_spec_in = {c: data[:, :, w//2-ks:, :] for c, data in spec_in.items()}

        tf_config = tf.ConfigProto(
            device_count={'GPU': 1}, allow_soft_placement=True)
        with tf.Session(config=tf_config) as sess:
            diff_restore_path = config['kpcn']['diff'].get('restore_path', '')
            spec_restore_path = config['kpcn']['spec'].get('restore_path', '')

            # initialize/restore
            sess.run(tf.group(
                tf.global_variables_initializer(), tf.local_variables_initializer()))
            if os.path.isfile(diff_restore_path + '.index'):
                self.diff_kpcn.restore(sess, diff_restore_path)
            if os.path.isfile(spec_restore_path + '.index'):
                self.spec_kpcn.restore(sess, spec_restore_path)

            l_diff_out = self.diff_kpcn.run(self, sess, l_diff_in)
            r_diff_out = self.diff_kpcn.run(self, sess, r_diff_in)
            l_spec_out = self.spec_kpcn.run(self, sess, l_spec_in)
            r_spec_out = self.spec_kpcn.run(self, sess, r_spec_in)

        # composite
        diff_out = np.zeros((h, w, 3))
        diff_out[:, :w//2, :] = l_diff_out[0, :, :w//2, :]
        diff_out[:, w//2:, :] = r_diff_out[0, :, w//2:, :]
        spec_out = np.zeros((h, w, 3))
        spec_out[:, :w//2, :] = l_spec_out[0, :, :w//2, :]
        spec_out[:, w//2:, :] = r_spec_out[0, :, w//2:, :]

        # postprocess
        diff_out = du.postprocess_diffuse(diff_out, input_buffers['albedo'], self.eps)
        spec_out = du.postprocess_specular(spec_out)

        # combine
        out = diff_out + spec_out
        out = du.clip_and_gamma_correct(out)

        _in = np.concatenate(
            [input_buffers['R'], input_buffers['G'], input_buffers['B']], axis=-1)
        _in = du.clip_and_gamma_correct(_in)

        # write and show
        out_path = os.path.join(out_dir, 'out_' + os.path.basename(im_path))
        imsave(out_path, out)
        print('[+] Wrote result to %s.' % out_path)
        du.show_multiple(_in, out, row_max=2, block_on_viz=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    denoiser = Denoiser(config)
    getattr(denoiser, config['op'])(config)
