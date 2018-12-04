import os
import re
import sys
import glob
import yaml
import time
import shutil
import random
import argparse
import numpy as np
from math import sqrt
import tensorflow as tf
from scipy.misc import imsave

from kpcn import DKPCN, CombinedModel
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

        train_diff     = config['kpcn']['diff']['train_include']
        train_spec     = config['kpcn']['spec']['train_include']

        if type(learning_rate) == str:
            learning_rate = eval(learning_rate)
        if type(max_epochs) == str:
            max_epochs = int(eval(max_epochs))

        # load data
        comb = train_diff and train_spec
        tf_buffers, init_ops = self.load_data(config, comb=comb)

        tf_config = tf.ConfigProto(
            device_count={'GPU': 1}, allow_soft_placement=True)
        with tf.Session(config=tf_config) as sess:
            # training loops

            if train_diff:
                diff_checkpoint_dir = os.path.join(checkpoint_dir, 'diff')
                diff_restore_path   = config['kpcn']['diff'].get('restore_path', '')
                _tf_buffers = tf_buffers['diff'] if comb else tf_buffers
                self.diff_kpcn = DKPCN(
                    _tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='diffuse')

            if train_spec:
                spec_checkpoint_dir = os.path.join(checkpoint_dir, 'spec')
                spec_restore_path   = config['kpcn']['spec'].get('restore_path', '')
                _tf_buffers = tf_buffers['spec'] if comb else tf_buffers
                self.spec_kpcn = DKPCN(
                    _tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='specular')

            if comb:
                self.comb_kpcn      = CombinedModel(
                    self.diff_kpcn, self.spec_kpcn, tf_buffers['comb'], self.eps, learning_rate, summary_dir)
                comb_checkpoint_dir = os.path.join(checkpoint_dir, 'comb')
                restore_path        = (diff_restore_path, spec_restore_path)
                self._training_loop(sess, self.comb_kpcn, init_ops['comb_train'], init_ops['comb_val'], 'comb',
                    log_freq, save_freq, viz_freq, max_epochs, comb_checkpoint_dir, block_on_viz, restore_path, reset_lr)
            elif train_diff:
                self._training_loop(sess, self.diff_kpcn, init_ops['diff_train'], init_ops['diff_val'], 'diff',
                    log_freq, save_freq, viz_freq, max_epochs, diff_checkpoint_dir, block_on_viz, diff_restore_path, reset_lr)
            elif train_spec:
                self._training_loop(sess, self.spec_kpcn, init_ops['spec_train'], init_ops['spec_val'], 'spec',
                    log_freq, save_freq, viz_freq, max_epochs, spec_checkpoint_dir, block_on_viz, spec_restore_path, reset_lr)

    def load_data(self, config, shuffle=True, comb=False):
        batch_size        = config['train_params'].get('batch_size', 5)
        patch_size        = config['data'].get('patch_size', 65)
        tfrecord_dir      = config['data']['tfrecord_dir']
        scenes            = config['data']['scenes']
        clip_ims          = config['data']['clip_ims']
        shuffle_filenames = config['data']['shuffle_filenames']

        train_filenames = []
        val_filenames   = []
        for scene in scenes:
            train_filenames.extend(glob.glob(os.path.join(tfrecord_dir, scene, 'train', 'data*.tfrecords')))
            val_filenames.extend(glob.glob(os.path.join(tfrecord_dir, scene, 'validation', 'data*.tfrecords')))

        def shuffled_dataset(filenames):
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.shuffle(len(filenames))
            return dataset.interleave(tf.data.TFRecordDataset, cycle_length=min(750, len(filenames)))

        with tf.device('/cpu:0'):
            if shuffle_filenames:
                train_dataset = shuffled_dataset(train_filenames)
                val_dataset   = shuffled_dataset(val_filenames)
            else:
                train_dataset = tf.data.TFRecordDataset(train_filenames)
                val_dataset   = tf.data.TFRecordDataset(val_filenames)

            pstep = config['data'].get('parse_step', 1)
            patches_per_im = config['data'].get('patches_per_im', 400)
            train_dataset_size_estimate = (int(200 / pstep) * patches_per_im) * len(train_filenames)
            val_dataset_size_estimate   = (int(200 / pstep) * patches_per_im) * len(val_filenames)
            size_lim = config['data'].get('shuffle_buffer_size_limit', 50000)

            if comb:
                decode_comb = du.make_decode('comb', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims)
                datasets = {
                    'comb_train': train_dataset.map(decode_comb, num_parallel_calls=4),
                    'comb_val': val_dataset.map(decode_comb, num_parallel_calls=4),
                }
            else:
                decode_diff = du.make_decode('diff', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims)
                decode_spec = du.make_decode('spec', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims)
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
                if shuffle:
                    datasets[comp] = datasets[comp].shuffle(dataset_size)
                datasets[comp] = datasets[comp].batch(batch_size)

            # shared iterator
            iterator = tf.data.Iterator.from_structure(
                datasets.values()[0].output_types, datasets.values()[0].output_shapes)
            if comb:
                diff, spec, normal, albedo, depth, var_diff, var_spec, \
                    var_features, gt_diff, gt_spec, gt_out = iterator.get_next()

                # initializers
                comb_train_init_op = iterator.make_initializer(datasets['comb_train'])
                comb_val_init_op   = iterator.make_initializer(datasets['comb_val'])

                # compute gradients
                grad = []
                grad_y_normal, grad_x_normal = tf.image.image_gradients(normal)
                grad_y_albedo, grad_x_albedo = tf.image.image_gradients(albedo)
                grad_y_depth, grad_x_depth = tf.image.image_gradients(depth)
                for color in (diff, spec):
                    grad_y_color, grad_x_color = tf.image.image_gradients(color)
                    grad_y = tf.concat([grad_y_color, grad_y_normal, grad_y_albedo, grad_y_depth], -1)
                    grad_x = tf.concat([grad_x_color, grad_x_normal, grad_x_albedo, grad_x_depth], -1)
                    grad.append({'grad_y': grad_y, 'grad_x': grad_x})

                tf_buffers_diff = {
                    'color': diff,
                    'grad_x': grad[0]['grad_x'],
                    'grad_y': grad[0]['grad_y'],
                    'var_color': var_diff,
                    'var_features': var_features,
                    'gt_out': gt_diff,
                }
                tf_buffers_spec = {
                    'color': spec,
                    'grad_x': grad[1]['grad_x'],
                    'grad_y': grad[1]['grad_y'],
                    'var_color': var_spec,
                    'var_features': var_features,
                    'gt_out': gt_spec,
                }
                tf_buffers_comb = {
                    'albedo': albedo,
                    'gt_out': gt_out,
                }
                tf_buffers = {
                    'diff': tf_buffers_diff,
                    'spec': tf_buffers_spec,
                    'comb': tf_buffers_comb,
                }
                init_ops = {
                    'comb_train': comb_train_init_op,
                    'comb_val': comb_val_init_op,
                }
            else:
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
        file_example_limit   = config['data'].get('file_example_limit', 1e5)

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
                if split == 'test' and splits['test'] != 0.0 and sum(splits.values()) == 1.0:
                    end = len(input_exr_files)
                print('[o] %s split: %d/%d permutations.' % (split, end - start, len(input_exr_files)))

                if end - start == 0:
                    continue

                tfrecord_scene_split_dir = os.path.join(tfrecord_scene_dir, split)
                if not os.path.exists(tfrecord_scene_split_dir):
                    os.makedirs(tfrecord_scene_split_dir)
                tfrecord_filepath = os.path.join(tfrecord_scene_split_dir, 'data%d.tfrecords')
                debug_dir = tfrecord_scene_split_dir if save_debug_ims else ''
                du.write_tfrecords(
                    tfrecord_filepath, input_exr_files[start:end], gt_exr_files[start:end],
                    patches_per_im, patch_size, self.fp16, shuffle=pshuffle, debug_dir=debug_dir,
                    save_debug_ims_every=save_debug_ims_every, color_var_weight=color_var_weight,
                    normal_var_weight=normal_var_weight, file_example_limit=file_example_limit)

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
        gt_spp        = config['data']['gt_spp']
        out_dir       = config['evaluate']['out_dir']

        if type(learning_rate) == str:
            learning_rate = eval(learning_rate)

        im_path = config['evaluate']['im_path']
        if im_path.endswith('exr'):
            input_buffers = du.read_exr(im_path, fp16=self.fp16)
            input_buffers = du.stack_channels(input_buffers)
        else:
            dot_idx      = im_path.rfind('.')
            filetype_str = ' ' + im_path[dot_idx:] if dot_idx != -1 else ''
            raise TypeError('[-] Filetype%s not supported yet.' % filetype_str)
        print('[o] Denoising %s...' % os.path.basename(im_path))

        h, w = input_buffers['diffuse'].shape[:2]
        ks = int(sqrt(layers_config[-1]['num_outputs']))

        # clip
        if config['data']['clip_ims']:
            input_buffers['diffuse'] = du.clip_and_gamma_correct(input_buffers['diffuse'])
            input_buffers['specular'] = du.clip_and_gamma_correct(input_buffers['specular'])
            # input_buffers['diffuse'] = np.clip(input_buffers['diffuse'], 0.0, 1.0)
            # input_buffers['specular'] = np.clip(input_buffers['specular'], 0.0, 1.0)

        # preprocess
        du.preprocess_diffuse(input_buffers, self.eps)
        du.preprocess_specular(input_buffers)
        du.preprocess_depth(input_buffers)

        # make network inputs
        diff_in, spec_in = du.make_network_inputs(input_buffers)

        # split image into l/r sub-images (with overlap)
        # because my GPU doesn't have enough memory to deal with the whole image at once
        l_diff_in = {c: data[:, :, :w//2+ks,  :] for c, data in diff_in.items()}
        r_diff_in = {c: data[:, :, -w//2-ks:, :] for c, data in diff_in.items()}
        l_spec_in = {c: data[:, :, :w//2+ks,  :] for c, data in spec_in.items()}
        r_spec_in = {c: data[:, :, -w//2-ks:, :] for c, data in spec_in.items()}

        # define networks
        tf_placeholders = {
            'color':        tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),
            'grad_x':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 10)),
            'grad_y':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 10)),
            'var_color':    tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 1)),
            'var_features': tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),
            'gt_out':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),  # not used
        }
        self.diff_kpcn = DKPCN(
            tf_placeholders, patch_size, patch_size, layers_config,
            is_training, learning_rate, summary_dir, scope='diffuse')
        self.spec_kpcn = DKPCN(
            tf_placeholders, patch_size, patch_size, layers_config,
            is_training, learning_rate, summary_dir, scope='specular')

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

            start_time = time.time()
            l_diff_out = self.diff_kpcn.run(sess, l_diff_in)
            r_diff_out = self.diff_kpcn.run(sess, r_diff_in)
            l_spec_out = self.spec_kpcn.run(sess, l_spec_in)
            r_spec_out = self.spec_kpcn.run(sess, r_spec_in)
            print('[o] graph execution time: %s' % du.format_seconds(time.time() - start_time))

        # composite
        diff_out = np.zeros((h, w, 3), dtype=np.float32)
        diff_out[:, :w//2, :] = l_diff_out[0, :, :w//2,  :]
        diff_out[:, w//2:, :] = r_diff_out[0, :, -w//2:, :]
        spec_out = np.zeros((h, w, 3), dtype=np.float32)
        spec_out[:, :w//2, :] = l_spec_out[0, :, :w//2,  :]
        spec_out[:, w//2:, :] = r_spec_out[0, :, -w//2:, :]

        # postprocess
        diff_out = du.postprocess_diffuse(diff_out, input_buffers['albedo'], self.eps)
        spec_out = du.postprocess_specular(spec_out)

        # combine
        _out = diff_out + spec_out
        out = du.clip_and_gamma_correct(_out)

        _in = np.concatenate(
            [input_buffers['R'], input_buffers['G'], input_buffers['B']], axis=-1)
        _in = du.clip_and_gamma_correct(_in)

        # try to get gt image for comparison (might not exist)
        _gt_out = None
        gt_out  = None
        match   = re.match(r'^(/.*\d+)-\d{5}spp.exr$', im_path)
        if match:
            gt_path = '%s-%sspp.exr' % (match.group(1), str(gt_spp).zfill(5))
            gt_buffers = du.read_exr(gt_path, fp16=self.fp16)
            gt_buffers = du.stack_channels(gt_buffers)
            _gt_out = np.concatenate(
                [gt_buffers['R'], gt_buffers['G'], gt_buffers['B']], axis=-1)
            gt_out = du.clip_and_gamma_correct(_gt_out)

        # write and show
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        _input_id = os.path.basename(im_path)
        _input_id = _input_id[:_input_id.rfind('.')]
        imsave(os.path.join(out_dir, _input_id + '_00_in.jpg'),  _in)
        imsave(os.path.join(out_dir, _input_id + '_01_out.jpg'), out)
        print('[+] Wrote result to %s.'
            % os.path.join(out_dir, _input_id + '_01_out.jpg'))
        ims_to_viz = [_in, out]
        if gt_out is not None:
            ims_to_viz.append(gt_out)
            imsave(os.path.join(out_dir, _input_id + '_02_gt.jpg'), gt_out)
        du.show_multiple(*ims_to_viz, row_max=len(ims_to_viz), block_on_viz=True)

        # write EXRs
        du.write_exr({
            'R': _out[:, :, 0],
            'G': _out[:, :, 1],
            'B': _out[:, :, 2]}, os.path.join(out_dir, _input_id + '_01_out.exr'))
        du.write_exr({
            'R': _gt_out[:, :, 0],
            'G': _gt_out[:, :, 1],
            'B': _gt_out[:, :, 2]}, os.path.join(out_dir, _input_id + '_02_gt.exr'))

        # write error images
        if gt_out is not None:
            # error WITHOUT clipping and gamma correction
            error = np.abs(_out - _gt_out)
            imsave(os.path.join(out_dir, _input_id + '_03_error.jpg'), error)
            diff_error = np.abs(diff_out - gt_buffers['diffuse'])
            imsave(os.path.join(out_dir, _input_id + '_04_diff_error.jpg'), diff_error)
            spec_error = np.abs(spec_out - gt_buffers['specular'])
            imsave(os.path.join(out_dir, _input_id + '_05_spec_error.jpg'), spec_error)

            # error WITH clipping and gamma correction
            perceptual_error = np.abs(out - gt_out)
            imsave(os.path.join(out_dir, _input_id + '_06_perceptual_error.jpg'), perceptual_error)
            perceptual_diff_error = np.abs(
                du.clip_and_gamma_correct(diff_out) - du.clip_and_gamma_correct(gt_buffers['diffuse']))
            imsave(os.path.join(out_dir, _input_id + '_07_perceptual_diff_error.jpg'), perceptual_diff_error)
            perceptual_spec_error = np.abs(
                du.clip_and_gamma_correct(spec_out) - du.clip_and_gamma_correct(gt_buffers['specular']))
            imsave(os.path.join(out_dir, _input_id + '_08_perceptual_spec_error.jpg'), perceptual_spec_error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    denoiser = Denoiser(config)
    getattr(denoiser, config['op'])(config)
