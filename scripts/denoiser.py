import os
import glob
import yaml
import time
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
        self.eps = 0.00316
        self.diff_kpcn = None
        self.spec_kpcn = None
        self.fp16 = config['train_params'].get('fp16', True)
        self.tf_dtype = tf.float16 if self.fp16 else tf.float32

    def train(self, config):
        log_freq = config['train_params'].get('log_freq', 100)
        save_freq = config['train_params'].get('save_freq', 1000)
        viz_freq = config['train_params'].get('viz_freq', -1)
        checkpoint_dir = config['train_params']['checkpoint_dir']
        max_epochs = config['train_params'].get('max_epochs', 1e9)
        batch_size = config['train_params'].get('batch_size', 5)
        patch_size = config['data'].get('patch_size', 65)
        layers_config = config['layers']
        is_training = config['op'] == 'train'
        learning_rate = config['train_params']['learning_rate']
        summary_dir = config['train_params'].get('summary_dir', 'summaries')
        block_on_viz = config['train_params'].get('block_on_viz', False)

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
            if config['kpcn']['incl_diff']:
                self.diff_kpcn = KPCN(
                    tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='diffuse')

                # initialization
                sess.run(tf.group(
                    tf.global_variables_initializer(), tf.local_variables_initializer()))

                i = 0
                for epoch in range(max_epochs):
                    # Training
                    sess.run(init_ops['diff_train'])
                    total_loss, count = 0.0, 0
                    while True:
                        try:
                            loss = self.diff_kpcn.run_train_step(sess, i)
                            total_loss += loss
                            count += batch_size
                            if (i + 1) % log_freq == 0:
                                print('[step %07d] diff loss: %.5f' % (i, total_loss / count))
                                total_loss, count = 0.0, 0
                            if (i + 1) % save_freq == 0:
                                self.diff_kpcn.save(sess, i, checkpoint_dir=os.path.join(checkpoint_dir, 'diff_kpcn'))
                                print('[o] Saved model.')
                        except tf.errors.OutOfRangeError:
                            break
                        i += 1
                    print('[o][diff] Epoch %d training complete. Running validation...' % (epoch + 1,))

                    # Validation
                    sess.run(init_ops['diff_val'])
                    total_loss, count = 0.0, 0
                    while True:
                        try:
                            loss, _in, _out, _gt = self.diff_kpcn.run_validation(sess)
                            if count == 0 and viz_freq > 0 and (i + 1) % viz_freq == 0:
                                du.show_multiple(_in, _out, _gt, block_on_viz=block_on_viz)
                            total_loss += loss
                            count += batch_size
                        except tf.errors.OutOfRangeError:
                            break
                    print('[o][diff] Validation loss: %.5f' % (total_loss / count,))
                    print('[o][diff] Epoch %d complete.' % (epoch + 1,))

            if config['kpcn']['incl_spec']:
                self.spec_kpcn = KPCN(
                    tf_buffers, patch_size, patch_size, layers_config,
                    is_training, learning_rate, summary_dir, scope='specular')

                # initialization
                sess.run(tf.group(
                    tf.global_variables_initializer(), tf.local_variables_initializer()))

                i = 0
                for epoch in range(max_epochs):
                    # Training
                    sess.run(init_ops['spec_train'])
                    total_loss, count = 0.0, 0
                    while True:
                        try:
                            loss = self.spec_kpcn.run_train_step(sess, i)
                            total_loss += loss
                            count += batch_size
                            if (i + 1) % log_freq == 0:
                                print('[step %07d] spec loss: %.5f' % (i, total_loss / count))
                                total_loss, count = 0.0, 0
                            if (i + 1) % save_freq == 0:
                                self.spec_kpcn.save(sess, i, checkpoint_dir=os.path.join(checkpoint_dir, 'spec_kpcn'))
                                print('[o] Saved model.')
                        except tf.errors.OutOfRangeError:
                            break
                        i += 1
                    print('[o][diff] Epoch %d training complete. Running validation...' % (epoch + 1,))

                    # Validation
                    sess.run(init_ops['spec_val'])
                    total_loss, count = 0.0, 0
                    while True:
                        try:
                            loss, _in, _out, _gt = self.spec_kpcn.run_validation(sess)
                            if count == 0 and viz_freq > 0 and (i + 1) % viz_freq == 0:
                                du.show_multiple(_in, _out, _gt, block_on_viz=block_on_viz)
                            total_loss += loss
                            count += batch_size
                        except tf.errors.OutOfRangeError:
                            break
                    print('[o][spec] Validation loss: %.5f' % (total_loss / count,))
                    print('[o][spec] Epoch %d complete.' % (epoch + 1,))

    def load_data(self, config):
        batch_size = config['train_params'].get('batch_size', 5)
        patch_size = config['data'].get('patch_size', 65)
        tfrecord_dir = config['data']['tfrecord_dir']
        scenes = config['data']['scenes']

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

            decode_diff = du.make_decode(True, self.tf_dtype, patch_size, patch_size, self.eps)
            decode_spec = du.make_decode(False, self.tf_dtype, patch_size, patch_size, self.eps)
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
        """Write data to TFRecords files for later use."""
        data_dir = config['data']['data_dir']
        scenes = config['data']['scenes']
        splits = config['data']['splits']
        in_filename = '*-%sspp.exr' % str(config['data']['in_spp']).zfill(5)
        gt_filename = '*-%sspp.exr' % str(config['data']['gt_spp']).zfill(5)

        tfrecord_dir = config['data']['tfrecord_dir']
        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)

        patch_size = config['data'].get('patch_size', 65)
        patches_per_im = config['data'].get('patches_per_im', 400)

        pstart = config['data'].get('parse_start', 0)
        pstep = config['data'].get('parse_step', 1)

        for scene in scenes:
            input_exr_files = sorted(glob.glob(os.path.join(data_dir, scene, in_filename)))
            gt_exr_files    = sorted(glob.glob(os.path.join(data_dir, scene, gt_filename)))
            # can skip files if on a time/memory budget
            input_exr_files = [f for i, f in enumerate(input_exr_files[pstart:]) if i % pstep == 0]
            gt_exr_files    = [f for i, f in enumerate(gt_exr_files[pstart:])    if i % pstep == 0]
            print('[o] Using %d permutations for scene %s.' % (len(input_exr_files), scene))

            end = 0
            for split in ['train', 'validation', 'test']:  # train gets the first 0.X, val gets the next 0.X, ...
                start = end  # [
                end = start + int(round(splits[split] * len(input_exr_files)))  # )
                if split == 'test':
                    end = len(input_exr_files)
                print('[o] %s split: %d/%d permutations.' % (split, end - start, len(input_exr_files)))

                tfrecord_scene_split_dir = os.path.join(tfrecord_dir, scene, split)
                if not os.path.exists(tfrecord_scene_split_dir):
                    os.makedirs(tfrecord_scene_split_dir)
                tfrecord_filepath = os.path.join(tfrecord_scene_split_dir, 'data.tfrecords')
                du.write_tfrecords(
                    tfrecord_filepath, input_exr_files[start:end],
                    gt_exr_files[start:end], patches_per_im, patch_size, self.fp16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    denoiser = Denoiser(config)
    getattr(denoiser, config['op'])(config)
