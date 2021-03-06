import os
import re
import sys
import glob
import yaml
import time
import shutil
import random
import pickle
import argparse
import numpy as np
from math import sqrt
from tqdm import tqdm
import tensorflow as tf
from scipy.misc import imsave
import tensorflow.contrib.tensorrt as trt

from models import DKPCN, MultiscaleModel, CombinedModel
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
    def _training_loop(sess, kpcn, train_init_op, val_init_op, identifier,
                       log_freq, save_freq, viz_freq, max_epochs, checkpoint_dir,
                       block_on_viz, restore_path='', reset_lr=True, save_best=False,
                       only_val=False, dropout_keep_prob=0.7, aux_summary_write_freq=-1, indiv_spp=-1):
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
                if not only_val:
                    # Training
                    sess.run(train_init_op)
                    total_loss, count = 0.0, 0
                    while True:
                        try:
                            loss, gnorm, merged_summaries = \
                                kpcn.run_train_step(sess, i, dropout_keep_prob)
                            total_loss += loss
                            count += 1
                            if (i + 1) % log_freq == 0:
                                avg_loss = total_loss / count
                                if avg_loss < min_avg_train_loss:
                                    min_avg_train_loss = avg_loss
                                print('[step %07d] %s loss: %.5f | gnorm: %.7f' \
                                    % (i, identifier, avg_loss, gnorm))
                                total_loss, count = 0.0, 0
                            if (i + 1) % save_freq == 0 and not save_best:
                                kpcn.save(sess, i, checkpoint_dir=checkpoint_dir)
                                print('[o] Saved model.')
                            if aux_summary_write_freq > 0 \
                                    and (i + 1) % aux_summary_write_freq == 0 \
                                    and merged_summaries is not None:
                                kpcn.train_writer.add_summary(merged_summaries, i)
                        except tf.errors.OutOfRangeError:
                            break
                        i += 1
                    print('[o][%s] Epoch %d training complete.' % (identifier, epoch + 1,))
                print('[o][%s] Running validation...' % identifier)

                # Validation
                sess.run(val_init_op)
                total_loss, count = 0.0, 0
                while True:
                    try:
                        loss, _in, _out, _gt = kpcn.run_validation(sess)
                        if indiv_spp > 0:
                            # Make suitable for visualization
                            _in = np.mean(_in, axis=1)
                        if (count == 0 or only_val) and viz_freq > 0 and (i + 1) % viz_freq == 0:
                            du.show_multiple(_in, _out, _gt, block_on_viz=block_on_viz)
                        total_loss += loss
                        count += 1
                    except tf.errors.OutOfRangeError:
                        break
                avg_loss = total_loss / count
                if avg_loss < min_avg_val_loss:
                    min_avg_val_loss = avg_loss
                    if save_best:
                        kpcn.save(sess, i, checkpoint_dir=checkpoint_dir)
                        print('[o] Saved model (best iteration: %d).' % i)
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
        save_best      = config['train_params'].get('save_best', False)
        valid_padding  = config['kpcn'].get('valid_padding', False)
        only_val       = config['train_params'].get('only_val', False)
        asum_wfreq     = config['train_params'].get('aux_summary_write_freq', -1)

        train_diff     = config['kpcn']['diff']['train_include']
        train_spec     = config['kpcn']['spec']['train_include']
        multiscale     = config['kpcn'].get('multiscale', False)
        single_network = config['kpcn'].get('single_network', False)

        is_indiv_samples = config['data'].get('is_indiv_samples', False)
        indiv_spp = config['data']['in_spp'] if is_indiv_samples else -1

        if type(learning_rate) == str:
            learning_rate = eval(learning_rate)
        if type(max_epochs) == str:
            max_epochs = int(eval(max_epochs))

        asymmetric_loss     = config['train_params'].get('asymmetric_loss', True)
        clip_by_global_norm = config['train_params'].get('clip_by_global_norm', False)
        dropout_keep_prob   = config['train_params'].get('dropout_keep_prob', 0.7)

        # load data
        comb = train_diff and train_spec
        tf_buffers, init_ops = self.load_data(config, comb=comb, single_network=single_network)

        tf_config = tf.ConfigProto(
            device_count={'GPU': 1}, allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            # training loops

            Model = DKPCN if not multiscale else MultiscaleModel

            if single_network:
                checkpoint_dir = os.path.join(checkpoint_dir, 'single')
                restore_path = config['kpcn'].get('single_restore_path', '')
                if multiscale:
                    restore_path = (
                        restore_path, config['kpcn'].get('single_multiscale_restore_path', ''))
                kpcn = Model(
                    tf_buffers, patch_size, patch_size, layers_config, is_training,
                    learning_rate, summary_dir, scope='single', save_best=save_best,
                    fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                    valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)
                self.diff_kpcn = self.spec_kpcn = kpcn
                self._training_loop(sess, kpcn, init_ops['train'], init_ops['val'],
                    'single', log_freq, save_freq, viz_freq, max_epochs, checkpoint_dir, block_on_viz,
                    restore_path, reset_lr, save_best, only_val, dropout_keep_prob, asum_wfreq, indiv_spp)
            else:
                if train_diff:
                    diff_checkpoint_dir = os.path.join(checkpoint_dir, 'diff')
                    diff_restore_path   = config['kpcn']['diff'].get('restore_path', '')
                    if multiscale:
                        diff_restore_path = (
                            diff_restore_path, config['kpcn']['diff'].get('multiscale_restore_path', ''))
                    _tf_buffers = tf_buffers['diff'] if comb else tf_buffers
                    self.diff_kpcn = Model(
                        _tf_buffers, patch_size, patch_size, layers_config, is_training,
                        learning_rate, summary_dir, scope='diffuse', save_best=save_best,
                        fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                        valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)

                if train_spec:
                    spec_checkpoint_dir = os.path.join(checkpoint_dir, 'spec')
                    spec_restore_path   = config['kpcn']['spec'].get('restore_path', '')
                    if multiscale:
                        spec_restore_path = (
                            spec_restore_path, config['kpcn']['spec'].get('multiscale_restore_path', ''))
                    _tf_buffers = tf_buffers['spec'] if comb else tf_buffers
                    self.spec_kpcn = Model(
                        _tf_buffers, patch_size, patch_size, layers_config, is_training,
                        learning_rate, summary_dir, scope='specular', save_best=save_best,
                        fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                        valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)

                if comb:
                    self.comb_kpcn      = CombinedModel(
                        self.diff_kpcn, self.spec_kpcn, tf_buffers['comb'], self.eps,
                        learning_rate, summary_dir, fp16=self.fp16, clip_by_global_norm=clip_by_global_norm)
                    comb_checkpoint_dir = os.path.join(checkpoint_dir, 'comb')
                    restore_path        = (diff_restore_path, spec_restore_path)
                    self._training_loop(sess, self.comb_kpcn, init_ops['comb_train'], init_ops['comb_val'],
                        'comb', log_freq, save_freq, viz_freq, max_epochs, comb_checkpoint_dir, block_on_viz,
                        restore_path, reset_lr, save_best, only_val, dropout_keep_prob, asum_wfreq, indiv_spp)
                elif train_diff:
                    self._training_loop(sess, self.diff_kpcn, init_ops['diff_train'], init_ops['diff_val'],
                        'diff', log_freq, save_freq, viz_freq, max_epochs, diff_checkpoint_dir, block_on_viz,
                        diff_restore_path, reset_lr, save_best, only_val, dropout_keep_prob, asum_wfreq, indiv_spp)
                elif train_spec:
                    self._training_loop(sess, self.spec_kpcn, init_ops['spec_train'], init_ops['spec_val'],
                        'spec', log_freq, save_freq, viz_freq, max_epochs, spec_checkpoint_dir, block_on_viz,
                        spec_restore_path, reset_lr, save_best, only_val, dropout_keep_prob, asum_wfreq, indiv_spp)

    def load_data(self, config, shuffle=True, comb=False, single_network=False):
        batch_size        = config['train_params'].get('batch_size', 5)
        patch_size        = config['data'].get('patch_size', 65)
        tfrecord_dir      = config['data']['tfrecord_dir']
        scenes            = config['data']['scenes']
        clip_ims          = config['data']['clip_ims']
        shuffle_filenames = config['data']['shuffle_filenames']

        is_indiv_samples = config['data'].get('is_indiv_samples', False)
        indiv_spp = config['data']['in_spp'] if is_indiv_samples else -1

        train_filenames = []
        val_filenames   = []

        all_tfrds = tfrecord_dir if type(tfrecord_dir) == list else [tfrecord_dir]
        for tfrd in all_tfrds:
            for scene in scenes:
                train_filenames.extend(glob.glob(os.path.join(tfrd, scene, 'train', 'data*.tfrecords')))
                val_filenames.extend(glob.glob(os.path.join(tfrd, scene, 'validation', 'data*.tfrecords')))

        def shuffled_dataset(filenames):
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.shuffle(len(filenames))
            return dataset.interleave(tf.data.TFRecordDataset, cycle_length=min(750, len(filenames)))

        def merge_datasets(dataset0, dataset1):
            """Source: https://stackoverflow.com/a/47344405."""
            return tf.data.Dataset.zip((dataset0, dataset1)).flat_map(
                lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
                    tf.data.Dataset.from_tensors(x1)))

        with tf.device('/cpu:0'):
            if shuffle and shuffle_filenames:
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
                decode_comb = du.make_decode('comb', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims, indiv_spp=indiv_spp)
                datasets = {
                    'comb_train': train_dataset.map(decode_comb, num_parallel_calls=4),
                    'comb_val': val_dataset.map(decode_comb, num_parallel_calls=4),
                }
            else:
                decode_diff = du.make_decode('diff', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims, indiv_spp=indiv_spp)
                decode_spec = du.make_decode('spec', self.tf_dtype, patch_size, patch_size, self.eps, clip_ims, indiv_spp=indiv_spp)
                datasets = {
                    'diff_train': train_dataset.map(decode_diff, num_parallel_calls=4),
                    'spec_train': train_dataset.map(decode_spec, num_parallel_calls=4),
                    'diff_val': val_dataset.map(decode_diff, num_parallel_calls=4),
                    'spec_val': val_dataset.map(decode_spec, num_parallel_calls=4),
                }
                if single_network:
                    datasets = {
                        'train': merge_datasets(datasets['diff_train'], datasets['spec_train']),
                        'val': merge_datasets(datasets['diff_val'], datasets['spec_val']),
                    }
            for comp in datasets:
                if comp.endswith('train'):
                    dataset_size = min(train_dataset_size_estimate, size_lim)
                else:
                    dataset_size = min(val_dataset_size_estimate, size_lim)
                if shuffle:
                    datasets[comp] = datasets[comp].shuffle(dataset_size)
                if indiv_spp > 0:
                    datasets[comp] = datasets[comp].apply(
                        tf.contrib.data.batch_and_drop_remainder(batch_size))  # fixed batch size
                else:
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
                if indiv_spp > 0:
                    def stacked_gradients(batched_tensor):
                        grad_y_tensors, grad_x_tensors = zip(
                            *[tf.image.image_gradients(batched_tensor[:, i]) for i in range(indiv_spp)])
                        return tf.stack(grad_y_tensors, axis=1), tf.stack(grad_x_tensors, axis=1)
                    grad_y_diff, grad_x_diff = stacked_gradients(diff)
                    grad_y_spec, grad_x_spec = stacked_gradients(spec)
                    grad_y_normal, grad_x_normal = stacked_gradients(normal)
                    grad_y_albedo, grad_x_albedo = stacked_gradients(albedo)
                    grad_y_depth, grad_x_depth = stacked_gradients(depth)
                else:
                    grad_y_diff, grad_x_diff = tf.image.image_gradients(diff)
                    grad_y_spec, grad_x_spec = tf.image.image_gradients(spec)
                    grad_y_normal, grad_x_normal = tf.image.image_gradients(normal)
                    grad_y_albedo, grad_x_albedo = tf.image.image_gradients(albedo)
                    grad_y_depth, grad_x_depth = tf.image.image_gradients(depth)

                for grad_y_color, grad_x_color in [(grad_y_diff, grad_x_diff), (grad_y_spec, grad_x_spec)]:
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
                if single_network:
                    init_ops = {
                        'train': iterator.make_initializer(datasets['train']),
                        'val':   iterator.make_initializer(datasets['val']),
                    }
                else:
                    init_ops = {
                        'diff_train': iterator.make_initializer(datasets['diff_train']),
                        'spec_train': iterator.make_initializer(datasets['spec_train']),
                        'diff_val':   iterator.make_initializer(datasets['diff_val']),
                        'spec_val':   iterator.make_initializer(datasets['spec_val']),
                    }

                # compute gradients
                if indiv_spp > 0:
                    def stacked_gradients(batched_tensor):
                        grad_y_tensors, grad_x_tensors = zip(
                            *[tf.image.image_gradients(batched_tensor[:, i]) for i in range(indiv_spp)])
                        return tf.stack(grad_y_tensors, axis=1), tf.stack(grad_x_tensors, axis=1)
                    grad_y_color, grad_x_color = stacked_gradients(color)
                    grad_y_normal, grad_x_normal = stacked_gradients(normal)
                    grad_y_albedo, grad_x_albedo = stacked_gradients(albedo)
                    grad_y_depth, grad_x_depth = stacked_gradients(depth)
                else:
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

            # TODO for full comparison
            full_comparison = False
            if indiv_spp > 0 and full_comparison:
                for tf_buffer_name in tf_buffers.keys():
                    if not tf_buffer_name.startswith('gt'):
                        tf_buffers[tf_buffer_name] = tf.reduce_mean(tf_buffers[tf_buffer_name], axis=1)

        return tf_buffers, init_ops

    def sample_data(self, config):
        """Write data to TFRecord files for later use."""
        data_dir                    = config['data']['data_dir']
        scenes                      = config['data']['scenes']
        splits                      = config['data']['splits']
        in_filename                 = '*-%sspp.exr' % str(config['data']['in_spp']).zfill(5)
        gt_filename                 = '*-%sspp.exr' % str(config['data']['gt_spp']).zfill(5)
        tfrecord_dir                = config['data']['tfrecord_dir']
        pstart                      = config['data'].get('parse_start', 0)
        pstep                       = config['data'].get('parse_step', 1)
        pshuffle                    = config['data'].get('parse_shuffle', False)
        patch_size                  = config['data'].get('patch_size', 65)
        patches_per_im              = config['data'].get('patches_per_im', 400)
        save_debug_ims              = config['data'].get('save_debug_ims', False)
        save_debug_ims_every        = config['data'].get('save_debug_ims_every', 1)
        color_var_weight            = config['data'].get('color_var_weight', 1.0)
        normal_var_weight           = config['data'].get('normal_var_weight', 1.0)
        file_example_limit          = config['data'].get('file_example_limit', 1e5)
        use_error_maps_for_sampling = config['data'].get('use_error_maps_for_sampling', False)
        out_dir                     = config['evaluate']['out_dir']

        is_indiv_samples = config['data'].get('is_indiv_samples', False)
        indiv_spp = config['data']['in_spp'] if is_indiv_samples else -1
        indiv_spp_data_dir = config['data'].get('indiv_spp_data_dir', None)
        save_integrated_patches = config['data'].get('save_integrated_patches', False)

        if type(tfrecord_dir) == list:
            tfrecord_dir = tfrecord_dir[0]

        # load error maps
        error_maps = None
        if use_error_maps_for_sampling:
            errtype = 'spec' if 'spec' in tfrecord_dir else 'diff'
            em_base = '%s_error.pickle' % errtype
            em_filepath = os.path.join(out_dir, em_base)
            with open(em_filepath, 'rb') as handle:
                error_maps = pickle.load(handle)

        for scene in scenes:
            if indiv_spp > 0:
                in_filename = 'sample_stratified*.exr'
                indiv_subdirs = sorted(next(os.walk(os.path.join(indiv_spp_data_dir, scene)))[1])
                input_exr_files = [sorted(
                    glob.glob(os.path.join(indiv_spp_data_dir, scene, subdir, in_filename)),
                    key=du.natural_keys) for subdir in indiv_subdirs]  # list of lists
            else:
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
                    normal_var_weight=normal_var_weight, file_example_limit=file_example_limit,
                    error_maps=error_maps, indiv_spp=indiv_spp, save_integrated_patches=save_integrated_patches)

    def visualize_data(self, config, all_features=False):
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
                if all_features:
                    _in, grad_x, grad_y, var_color, var_features, _gt = sess.run([
                        tf_buffers['color'], tf_buffers['grad_x'], tf_buffers['grad_y'],
                        tf_buffers['var_color'], tf_buffers['var_features'], tf_buffers['gt_out']])

                    grad_x_color  = grad_x[0, :, :, :3]
                    grad_x_normal = grad_x[0, :, :, 3:6]
                    grad_x_albedo = grad_x[0, :, :, 6:9]
                    grad_x_depth  = grad_x[0, :, :, 9]

                    grad_y_color  = grad_y[0, :, :, :3]
                    grad_y_normal = grad_y[0, :, :, 3:6]
                    grad_y_albedo = grad_y[0, :, :, 6:9]
                    grad_y_depth  = grad_y[0, :, :, 9]

                    var_color  = var_color[0, :, :, 0]
                    var_normal = var_features[0, :, :, 0]
                    var_albedo = var_features[0, :, :, 1]
                    var_depth  = var_features[0, :, :, 2]

                    _in = _in[0, :, :, :]
                    _gt = _gt[0, :, :, :]

                    du.show_multiple(
                        _in, grad_x_color, grad_x_normal, grad_x_albedo, grad_x_depth,
                        grad_y_color, grad_y_normal, grad_y_albedo, grad_y_depth,
                        var_color, var_normal, var_albedo, var_depth, _gt, row_max=14, block_on_viz=True)
                else:
                    _in, _gt = sess.run([tf_buffers['color'], tf_buffers['gt_out']])
                    du.show_multiple(_in, _gt, block_on_viz=True)
                itr_remaining -= 1
                if itr_remaining == 0:
                    try:
                        itr_remaining = int(raw_input('Continue for X iterations: ').strip())
                    except ValueError:
                        print('[o] invalid response (must be integer), exiting...')
                        break

    def denoise(self, config, compute_error=False):
        """Run full pipeline to denoise an image.
        OR just compute error maps, if COMPUTE_ERROR is True.
        """
        patch_size      = config['data'].get('patch_size', 65)
        layers_config   = config['layers']
        is_training     = config['op'] == 'train'
        learning_rate   = config['train_params']['learning_rate']
        summary_dir     = config['train_params'].get('summary_dir', 'summaries')
        gt_spp          = config['data']['gt_spp']
        out_dir         = config['evaluate']['out_dir']
        write_error_ims = config['evaluate']['write_error_ims']
        viz_kernels     = config['evaluate'].get('viz_kernels', False)
        clip_ims        = config['data']['clip_ims']
        valid_padding   = config['kpcn'].get('valid_padding', False)
        multiscale      = config['kpcn'].get('multiscale', False)
        use_trt         = config['evaluate'].get('use_trt', False)
        diff_frozen     = config['kpcn']['diff'].get('frozen', '')
        spec_frozen     = config['kpcn']['spec'].get('frozen', '')
        single_network  = config['kpcn'].get('single_network', False)

        is_indiv_samples = config['data'].get('is_indiv_samples', False)
        indiv_spp = config['data']['in_spp'] if is_indiv_samples else -1

        if type(learning_rate) == str:
            learning_rate = eval(learning_rate)

        asymmetric_loss     = config['train_params'].get('asymmetric_loss', True)
        clip_by_global_norm = config['train_params'].get('clip_by_global_norm', False)

        tf_config_kwargs = {
            'device_count': {'GPU': 1},
            'allow_soft_placement': True,
            'gpu_options': tf.GPUOptions(per_process_gpu_memory_fraction=0.75),
        }
        if use_trt:
            tf_config_kwargs['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        tf_config = tf.ConfigProto(**tf_config_kwargs)
        with tf.Session(config=tf_config) as sess:
            if compute_error:
                data_dir    = config['data']['data_dir']
                scenes      = config['data']['scenes']
                in_filename = '*-%sspp.exr' % str(config['data']['in_spp']).zfill(5)
                pstart      = config['data'].get('parse_start', 0)
                pstep       = config['data'].get('parse_step', 1)
                for scene in scenes:
                    input_exr_files = sorted(glob.glob(os.path.join(data_dir, scene, in_filename)))
                    input_exr_files = [f for i, f in enumerate(input_exr_files[pstart:]) if i % pstep == 0]
                im_paths = input_exr_files
            else:
                im_paths = [config['evaluate']['im_path']]

            # loop over all images (might just be one)
            for im_path in im_paths:
                if im_path.endswith('exr'):
                    input_buffers = du.read_exr(im_path, fp16=self.fp16)
                else:
                    dot_idx      = im_path.rfind('.')
                    filetype_str = ' ' + im_path[dot_idx:] if dot_idx != -1 else ''
                    raise TypeError('[-] Filetype%s not supported yet.' % filetype_str)
                print('[o] Denoising %s...' % os.path.basename(im_path))

                # (temp) for smaller images
                # du.crop_buffers(input_buffers, 110, -110, 160, -560)
                # du.crop_buffers(input_buffers, 0, 381, 0, 660)

                h, w = input_buffers['diffuse'].shape[:2]
                ks = int(sqrt(layers_config[-1]['num_outputs']))

                # make network inputs
                diff_in, spec_in = du.make_network_inputs(input_buffers, clip_ims, self.eps, indiv_spp=indiv_spp)

                # split image into l/r sub-images (with overlap)
                # because my GPU doesn't have enough memory to deal with the whole image at once
                l_diff_in = {c: data[:, :, :w//2+ks,  :] for c, data in diff_in.items()}
                r_diff_in = {c: data[:, :, -w//2-ks:, :] for c, data in diff_in.items()}
                l_spec_in = {c: data[:, :, :w//2+ks,  :] for c, data in spec_in.items()}
                r_spec_in = {c: data[:, :, -w//2-ks:, :] for c, data in spec_in.items()}

                if use_trt:
                    # see separate file (temp)
                    pass
                else:
                    if self.diff_kpcn is None:
                        # define networks
                        tf_placeholders = {
                            'color':        tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),
                            'grad_x':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 10)),
                            'grad_y':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 10)),
                            'var_color':    tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 1)),
                            'var_features': tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),
                            # not used
                            'gt_out':       tf.placeholder(self.tf_dtype, shape=(None, h, w // 2 + ks, 3)),
                        }
                        Model = DKPCN if not multiscale else MultiscaleModel
                        if single_network:
                            kpcn = Model(
                                tf_placeholders, patch_size, patch_size, layers_config,
                                is_training, learning_rate, summary_dir, scope='single',
                                fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                                valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)
                            restore_path = config['kpcn'].get('single_restore_path', '')
                            if multiscale:
                                restore_path = (
                                    restore_path, config['kpcn'].get('single_multiscale_restore_path', ''))
                            # initialize/restore
                            sess.run(tf.group(
                                tf.global_variables_initializer(), tf.local_variables_initializer()))
                            kpcn.restore(sess, restore_path)
                            self.diff_kpcn = self.spec_kpcn = kpcn
                        else:
                            self.diff_kpcn = Model(
                                tf_placeholders, patch_size, patch_size, layers_config,
                                is_training, learning_rate, summary_dir, scope='diffuse',
                                fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                                valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)
                            self.spec_kpcn = Model(
                                tf_placeholders, patch_size, patch_size, layers_config,
                                is_training, learning_rate, summary_dir, scope='specular',
                                fp16=self.fp16, clip_by_global_norm=clip_by_global_norm,
                                valid_padding=valid_padding, asymmetric_loss=asymmetric_loss, sess=sess, indiv_spp=indiv_spp)
                            diff_restore_path = config['kpcn']['diff'].get('restore_path', '')
                            spec_restore_path = config['kpcn']['spec'].get('restore_path', '')
                            if multiscale:
                                diff_restore_path = (
                                    diff_restore_path, config['kpcn']['diff'].get('multiscale_restore_path', ''))
                                spec_restore_path = (
                                    spec_restore_path, config['kpcn']['spec'].get('multiscale_restore_path', ''))

                            # initialize/restore
                            sess.run(tf.group(
                                tf.global_variables_initializer(), tf.local_variables_initializer()))
                            self.diff_kpcn.restore(sess, diff_restore_path)
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

                _in = du.clip_and_gamma_correct(input_buffers['default'])

                # try to get gt image for comparison (might not exist)
                _gt_out = None
                gt_out  = None
                match   = re.match(r'^(/.*\d+)-\d{5}spp.exr$', im_path)
                if match:
                    gt_path = '%s-%sspp.exr' % (match.group(1), str(gt_spp).zfill(5))
                    gt_buffers = du.read_exr(gt_path, fp16=self.fp16)

                    # (temp) for smaller images
                    # du.crop_buffers(gt_buffers, 110, -110, 160, -560)
                    # du.crop_buffers(gt_buffers, 0, 381, 0, 660)

                    _gt_out = gt_buffers['default']  # unmodified
                    gt_out = du.clip_and_gamma_correct(_gt_out)  # clipped and corrected

                    in_mse = du.mse(input_buffers['default'], _gt_out)
                    in_mrse = du.mrse(input_buffers['default'], _gt_out)
                    in_dssim = du.dssim(input_buffers['default'], _gt_out)
                    out_mse = du.mse(_out, _gt_out)
                    out_mrse = du.mrse(_out, _gt_out)
                    out_dssim = du.dssim(_out, _gt_out)
                    print('[o][input]  MSE: %.7f | MrSE: %.7f | DSSIM: %.7f' % (in_mse, in_mrse, in_dssim))
                    print('[o][output] MSE: %.7f | MrSE: %.7f | DSSIM: %.7f' % (out_mse, out_mrse, out_dssim))

                if compute_error:
                    diff_error = np.abs(diff_out - gt_buffers['diffuse'])
                    spec_error = np.abs(spec_out - gt_buffers['specular'])
                    du.write_error_map(diff_error, im_path, os.path.join(out_dir, 'diff_error.pickle'))
                    du.write_error_map(spec_error, im_path, os.path.join(out_dir, 'spec_error.pickle'))
                else:
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
                    du.write_exr(_out,    os.path.join(out_dir, _input_id + '_01_out.exr'))
                    du.write_exr(_gt_out, os.path.join(out_dir, _input_id + '_02_gt.exr'))

                    # write error images
                    if write_error_ims and gt_out is not None:
                        # error WITH clipping and gamma correction
                        perceptual_error = np.abs(out - gt_out)
                        imsave(os.path.join(out_dir, _input_id + \
                            '_03_perceptual_error.jpg'), perceptual_error)
                        perceptual_diff_error = np.abs(
                            du.clip_and_gamma_correct(diff_out) - \
                            du.clip_and_gamma_correct(gt_buffers['diffuse']))
                        imsave(os.path.join(out_dir, _input_id + \
                            '_04_perceptual_diff_error.jpg'), perceptual_diff_error)
                        perceptual_spec_error = np.abs(
                            du.clip_and_gamma_correct(spec_out) - \
                            du.clip_and_gamma_correct(gt_buffers['specular']))
                        imsave(os.path.join(out_dir, _input_id + \
                            '_05_perceptual_spec_error.jpg'), perceptual_spec_error)

                if viz_kernels and not use_trt:
                    # get kernels for 6x6 region at random location
                    y = np.random.randint(3 + ks // 2, h - 3 - ks // 2)
                    x = np.random.randint(3 + ks // 2, w - 3 - ks // 2)
                    if x < w // 2:
                        _diff_in = l_diff_in
                        _spec_in = l_spec_in
                    else:
                        _diff_in = r_diff_in
                        _spec_in = r_spec_in

                    diff_kernels = self.diff_kpcn.run(sess, _diff_in, self.diff_kpcn.out_kernels)[0]
                    if x < w // 2:
                        diff_kernels = diff_kernels[y-3:y+3, x-3:x+3, :]
                    else:
                        diff_kernels = diff_kernels[y-3:y+3, -(w-x)-3:-(w-x)+3, :]
                    diff_kernels = np.reshape(diff_kernels, (-1, ks, ks))  # (6 * 6, ks, ks)
                    diff_kernels = du.clip_and_gamma_correct(diff_kernels)

                    spec_kernels = self.spec_kpcn.run(sess, _spec_in, self.spec_kpcn.out_kernels)[0]
                    if x < w // 2:
                        spec_kernels = spec_kernels[y-3:y+3, x-3:x+3, :]
                    else:
                        spec_kernels = spec_kernels[y-3:y+3, -(w-x)-3:-(w-x)+3, :]
                    spec_kernels = np.reshape(spec_kernels, (-1, ks, ks))  # (6 * 6, ks, ks)
                    spec_kernels = du.clip_and_gamma_correct(spec_kernels)

                    rgb = du.clip_and_gamma_correct(input_buffers['default'])
                    rgb_highlight = rgb * 0.3
                    rgb_highlight[y-3:y+3, x-3:x+3, :] = rgb[y-3:y+3, x-3:x+3, :] * 2.0
                    rgb_highlight = rgb_highlight[
                        max(y-20,0):min(y+20,h), max(x-20,0):min(x+20,rgb.shape[1]), :]

                    du.show_multiple(rgb_highlight, row_max=1, block_on_viz=True)
                    du.show_multiple(*diff_kernels, block_on_viz=True)
                    du.show_multiple(*spec_kernels, block_on_viz=True)

    def compute_error(self, config):
        self.denoise(config, compute_error=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    denoiser = Denoiser(config)
    getattr(denoiser, config['op'])(config)
