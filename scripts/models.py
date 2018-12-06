import os
from math import sqrt
import tensorflow as tf
from numbers import Number

from data_utils import get_run_dir, tf_center_crop
from data_utils import tf_postprocess_diffuse, tf_postprocess_specular, tf_nan_to_num

class DKPCN(object):
    """
    Reconstruction network.

    - Direct prediction convolutional network, or
    - Kernel-predicting convolutional network, as described in Bako et al. 2017.

    The same graph definition is used for both the diffuse and specular networks.
    """
    curr_index = -1

    def __init__(self, tf_buffers, buffer_h, buffer_w, layers_config,
                 is_training, learning_rate, summary_dir, scope=None,
                 save_best=False, fp16=False, clip_by_global_norm=False,
                 valid_padding=False, asymmetric_loss=True, sess=None, reuse=False):

        self.buffer_h = buffer_h
        self.buffer_w = buffer_w
        self.is_training = is_training
        self.valid_padding = valid_padding and self.is_training

        if scope is None:
            DKPCN.curr_index += 1
        self.scope = scope or 'DKPCN_%d' % DKPCN.curr_index
        with tf.variable_scope(self.scope, reuse=reuse):
            with tf.variable_scope('inputs'):
                # color buffer
                self.color = tf.identity(tf_buffers['color'], name='color')  # (?, h, w, 3)
                # gradients (color, surface normals, albedo, depth)
                self.grad_x = tf.identity(tf_buffers['grad_x'], name='grad_x')  # (?, h, w, 10)
                self.grad_y = tf.identity(tf_buffers['grad_y'], name='grad_y')  # (?, h, w, 10)
                # rel variance
                self.var_color = tf.identity(tf_buffers['var_color'], name='var_color')  # (?, h, w, 1)
                self.var_features = tf.identity(tf_buffers['var_features'], name='var_features')  # (?, h, w, 3)

            out = tf.concat((
                self.color, self.grad_x, self.grad_y, self.var_color, self.var_features), axis=3)
            out = tf_nan_to_num(out)

            i = 0
            for layer in layers_config:
                kernel_init = None
                if layer.get('kernel_init', None) == 'xavier':
                    kernel_init = tf.contrib.layers.xavier_initializer(seed=23)
                try:
                    activation = getattr(tf.nn, layer.get('activation', ''))
                except AttributeError:
                    activation = None
                chain = layer.get('chain', 1)
                for j in range(chain):
                    with tf.variable_scope('layer%d' % (i + j)):
                        if layer['type'] == 'conv2d':
                            padding = 'valid' if self.valid_padding else 'same'
                            out = tf.layers.conv2d(
                                out, layer['num_outputs'], layer['kernel_size'],
                                strides=layer['stride'], padding=padding, activation=activation,
                                kernel_initializer=kernel_init, name='conv2d')
                        elif layer['type'] == 'residual_block':
                            dropout_keep_prob = layer.get('dropout_keep_prob', None)
                            out = self._residual_block(
                                out, dropout_keep_prob, self.is_training, kernel_init,
                                layer.get('num_outputs', 100), layer.get('kernel_size', 3))
                        elif layer['type'] == 'batch_normalization':
                            out = tf.layers.batch_normalization(
                                out, training=self.is_training, name='batch_normalization')
                        else:
                            raise ValueError('unsupported DKPCN layer type')
                i += chain

            if layer['num_outputs'] == 3:
                # DPCN
                self.kernel_size = None
                self.out_kernels = None
                _, self.valid_h, self.valid_w, _ = out.get_shape().as_list()
                self.out = out
            else:
                # KPCN
                self.kernel_size = int(sqrt(layer['num_outputs']))
                out_max =  tf.reduce_max(out, axis=-1, keepdims=True)
                self.out_kernels = tf.nn.softmax(out - out_max, axis=-1)
                _, self.valid_h, self.valid_w, _ = self.out_kernels.get_shape().as_list()
                self.out = self._filter(self.color, self.out_kernels)  # filtered color buffer

            # loss
            gt_out = tf_buffers['gt_out']
            if self.valid_padding:
                gt_out = tf_center_crop(gt_out, self.valid_h, self.valid_w)
            self.gt_out = tf.identity(gt_out, name='gt_out')  # (?, h, w, 3)
            if asymmetric_loss:
                loss = self._asymmetric_smape(self.color, self.out, self.gt_out, slope=2.0)
            else:
                loss = self._smape(self.out, self.gt_out)
            loss = tf.verify_tensor_all_finite(loss, 'NaN or Inf in loss')
            self.loss = tf.identity(loss, name='loss')
            self.loss_summary = tf.summary.scalar('loss', self.loss)

            # optimization
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100000, 0.96)
            if fp16:
                # eps: https://stackoverflow.com/a/42077538
                opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)
            else:
                opt = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = opt.compute_gradients(self.loss)  # [(grad, var) tuples]
            grads_and_vars = filter(lambda gv: None not in gv, grads_and_vars)
            grads, _vars = zip(*grads_and_vars)
            self.grads = grads  # list of gradients
            self.tvars = _vars  # list of trainable variables
            self.gnorm = tf.global_norm(grads, name='grad_norm')
            if clip_by_global_norm:  # empirically not helpful
                grads, _ = tf.clip_by_global_norm(grads, 0.5, use_norm=self.gnorm)
                grads_and_vars = zip(grads, _vars)
            # clip by value
            grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
            self.opt_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)

            # logging
            graph = sess.graph if sess is not None else None
            self.train_writer = tf.summary.FileWriter(
                get_run_dir(os.path.join(summary_dir, 'train')), graph)
            self.validation_writer = tf.summary.FileWriter(
                get_run_dir(os.path.join(summary_dir, 'validation')))
            mtk = 1 if save_best else 5
            self.saver = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope), max_to_keep=mtk)

    def _filter(self, color, kernels):

        # `color`   : (?, h, w, 3)
        # `kernels` : (?, h, w, kernel_size * kernel_size)

        _, im_h, im_w, _ = color.get_shape().as_list()    # all locns
        # (self.valid_h, self.valid_w): locations we have kernels for

        kernel_radius = self.kernel_size // 2
        y_extent = self.valid_h + kernel_radius * 2
        x_extent = self.valid_w + kernel_radius * 2  # add an extra kernel_radius on each side
        if y_extent > im_h:
            # zero-pad color buffer
            pad_y = int((y_extent - im_h) / 2)
            pad_x = int((x_extent - im_w) / 2)
            color = tf.pad(color, [[0, 0], [pad_y, pad_y], [pad_x, pad_x], [0, 0]])
        else:
            color = tf_center_crop(color, y_extent, x_extent)

        # filter channels separately
        filtered_channels = []
        for c in range(3):
            color_channel = color[:, :, :, c]
            color_channel = tf.expand_dims(color_channel, -1)
            color_channel = tf.extract_image_patches(
                color_channel, ksizes=[1, self.kernel_size, self.kernel_size, 1],
                strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')  # (?, h, w, kernel_size * kernel_size)

            # apply per-pixel kernels
            filtered_channels.append(
                tf.reduce_sum(color_channel * kernels, axis=-1))

        return tf.stack(filtered_channels, axis=3)  # (?, h, w, 3)

    @staticmethod
    def _l1_loss(out, gt_out):
        return tf.reduce_mean(tf.abs(out - gt_out))

    @staticmethod
    def _smape(out, gt_out):
        """Symmetric mean absolute percentage error."""
        return tf.reduce_mean(tf.abs(out - gt_out) / (tf.abs(out) + tf.abs(gt_out) + 1e-2))

    @staticmethod
    def _asymmetric_smape(_in, out, gt_out, slope):
        ape = tf.abs(out - gt_out) / (tf.abs(out) + tf.abs(gt_out) + 1e-2)
        return tf.reduce_mean(
            ape * (1.0 + (slope - 1.0) * DKPCN._heaviside((out - gt_out) * (gt_out - _in))))

    @staticmethod
    def _heaviside(x):
        """Elementwise Heaviside step function."""
        return tf.maximum(tf.sign(x), 0.0)

    @staticmethod
    def _residual_block(_in, dropout_keep_prob, is_training,
                        kernel_init=None, num_outputs=100, kernel_size=3):
        with tf.variable_scope('residual_block'):
            out = _in
            for k in range(2):
                out = tf.layers.conv2d(
                    out, filters=num_outputs, kernel_size=kernel_size, strides=1,
                    padding='same', activation=None, kernel_initializer=kernel_init)
                with tf.variable_scope("batchnorm%d" % k):
                    out = tf.layers.batch_normalization(
                        out, training=is_training, name='batch_normalization')
                out = tf.nn.relu(out)
                if isinstance(dropout_keep_prob, Number):
                    out = tf.layers.dropout(out, 1.0 - dropout_keep_prob, training=is_training)
        return _in + out

    def run(self, sess, batched_buffers, tensor=None):
        feed_dict = {
            self.color: batched_buffers['color'],
            self.grad_x: batched_buffers['grad_x'],
            self.grad_y: batched_buffers['grad_y'],
            self.var_color: batched_buffers['var_color'],
            self.var_features: batched_buffers['var_features'],
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration):
        _, loss, loss_summary, gnorm = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm])
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm

    def run_validation(self, sess):
        return sess.run([self.loss, self.color, self.out, self.gt_out])

    def save(self, sess, iteration, checkpoint_dir='checkpoints', write_meta_graph=True):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        base_filepath = os.path.join(checkpoint_dir, 'var')
        self.saver.save(sess, base_filepath, global_step=iteration, write_meta_graph=write_meta_graph)
        print('[+] Saved current parameters to %s-%d.' % (base_filepath, iteration))

    @staticmethod
    def _optimistic_restore(sess, restore_path):
        """Source: https://github.com/tensorflow/tensorflow/issues/312."""
        reader = tf.train.NewCheckpointReader(restore_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
            for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], 
            tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, restore_path)

    def restore(self, sess, restore_path, optimistic=False):
        if os.path.isfile(restore_path + '.index'):
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
            if optimistic:
                self._optimistic_restore(sess, restore_path)  # ignore vars not in checkpoint
            else:
                saver.restore(sess, restore_path)
            print('[+] `%s` network restored from `%s`.' % (self.scope, restore_path))

class MultiscaleModel(object):
    """Wrapper around a standard KPCN which combines results from three different scales."""

    def __init__(self, tf_buffers, buffer_h, buffer_w, *args, **kwargs):

        tf_buffers2 = self._resize_all(tf_buffers, buffer_h // 2, buffer_w // 2)
        tf_buffers4 = self._resize_all(tf_buffers, buffer_h // 4, buffer_w // 4)

        # fine to coarse
        kwargs['reuse'] = True
        self.kpcn1 = DKPCN(tf_buffers, buffer_h, buffer_w, *args, **kwargs)
        self.kpcn2 = DKPCN(tf_buffers2, buffer_h // 2, buffer_w // 2, *args, **kwargs)
        self.kpcn4 = DKPCN(tf_buffers4, buffer_h // 4, buffer_w // 4, *args, **kwargs)

        self.denoised1 = tf.stop_gradient(self.kpcn1.out)
        self.denoised2 = tf.stop_gradient(self.kpcn2.out)
        self.denoised4 = tf.stop_gradient(self.kpcn4.out)

        self.out = self._scale_compositor(self.denoised2, self.denoised4)
        self.out = self._scale_compositor(self.denoised1, self.out)  # reuse vars, so same as first compositor

        self.color = self.kpcn1.color
        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='scale_compositor')
        self.saver = tf.train.Saver(self.tvars, max_to_keep=5)

        # optimization
        # TODO

    def run(self, sess, batched_buffers, tensor=None):
        # TODO
        feed_dict = {
            self.color: batched_buffers['color'],
            self.grad_x: batched_buffers['grad_x'],
            self.grad_y: batched_buffers['grad_y'],
            self.var_color: batched_buffers['var_color'],
            self.var_features: batched_buffers['var_features'],
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration):
        # TODO
        _, loss, loss_summary, gnorm = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm])
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm

    def run_validation(self, sess):
        # TODO
        return sess.run([self.loss, self.color, self.out, self.gt_out])

    def restore(self, sess, restore_path, optimistic=False):
        self.kpcn1.restore(sess, restore_path, optimistic)  # reuse vars, so should restore all

    def _scale_compositor(self, denoised_fine, denoised_coarse):
        h, w = MultiscaleModel._get_spatial_dims(im)
        add_one = h % 2 == 1
        with tf.variable_scope('scale_compositor', reuse=True):
            alpha = tf.concat((denoised_fine, upsample2(denoised_coarse, add_one)), axis=-1)
            alpha = tf.layers.conv2d(
                alpha, filters=100, kernel_size=1, strides=1, padding='same', activation=None)
            alpha = self._residual_block(alpha, 0.9, self.is_training)
            alpha = self._residual_block(alpha, 0.9, self.is_training)
            alpha = tf.layers.conv2d(
                alpha, filters=1, kernel_size=1, strides=1, padding='same', activation=None)
            alpha = tf.nn.sigmoid(alpha)
            # blend fine and coarse images
            return denoised_fine \
                - alpha * upsample2(downsample2(denoised_fine), add_one) \
                + alpha * upsample2(denoised_coarse, add_one)

    @staticmethod
    def _resize_all(buffers, new_height, new_width):
        """Resize image tensors in the BUFFERS dictionary to (new_height, new_width)."""
        return {k: tf.image.resize_images(v,
            [new_height, new_width], method=ResizeMethod.BILINEAR) for k, v in buffers}

    @staticmethod
    def _downsample2(im):
        h, w = MultiscaleModel._get_spatial_dims(im)
        return tf.image.resize_images(im, [h // 2, w // 2], method=ResizeMethod.BILINEAR)

    @staticmethod
    def _upsample2(im, add_one=False):
        h, w = MultiscaleModel._get_spatial_dims(im)
        new_height, new_width = h * 2, w * 2
        if add_one:
            new_height += 1
            new_width  += 1
        return tf.image.resize_images(
            im, [new_height, new_width], method=ResizeMethod.NEAREST_NEIGHBOR)

    @staticmethod
    def _get_spatial_dims(im):
        im_shape = im.get_shape().as_list()
        if len(im_shape) == 4:
            h, w = im_shape[1], im_shape[2]
        else:
            h, w = im_shape[0], im_shape[1]
        return h, w

class CombinedModel(object):
    """Diffuse + specular KPCNs."""

    def __init__(self, diff_kpcn, spec_kpcn, tf_buffers_comb, eps,
                 learning_rate, summary_dir, clip_by_global_norm=False):

        self.diff_kpcn = diff_kpcn
        self.spec_kpcn = spec_kpcn

        self.albedo = tf_buffers_comb['albedo']
        self.gt_out = tf_buffers_comb['gt_out']

        if self.diff_kpcn.valid_padding:
            self.albedo = tf_center_crop(self.albedo, self.diff_kpcn.valid_h, self.diff_kpcn.valid_w)
            self.gt_out = tf_center_crop(self.gt_out, self.diff_kpcn.valid_h, self.diff_kpcn.valid_w)

        diff_out = tf_postprocess_diffuse(self.diff_kpcn.out, self.albedo, eps)
        spec_out = tf_postprocess_specular(self.spec_kpcn.out)
        self.out = diff_out + spec_out

        self.loss = tf.reduce_mean(tf.abs(self.out - self.gt_out), name='l1_loss')
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100000, 0.96)
        opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = filter(lambda gv: None not in gv, grads_and_vars)
        grads, _vars = zip(*grads_and_vars)
        self.gnorm = tf.global_norm(grads, name='grad_norm')
        if clip_by_global_norm:  # empirically not helpful
            grads, _ = tf.clip_by_global_norm(grads, 0.5, use_norm=self.gnorm)
            grads_and_vars = zip(grads, _vars)
        # clip by value
        grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
        self.opt_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.train_writer = tf.summary.FileWriter(
            get_run_dir(os.path.join(summary_dir, 'train')))
        self.validation_writer = tf.summary.FileWriter(
            get_run_dir(os.path.join(summary_dir, 'validation')))

    def run(self, sess, diff_batched_buffers, spec_batched_buffers, batched_albedo, tensor=None):
        feed_dict = {
            self.diff_kpcn.color:        diff_batched_buffers['color'],
            self.diff_kpcn.grad_x:       diff_batched_buffers['grad_x'],
            self.diff_kpcn.grad_y:       diff_batched_buffers['grad_y'],
            self.diff_kpcn.var_color:    diff_batched_buffers['var_color'],
            self.diff_kpcn.var_features: diff_batched_buffers['var_features'],
            self.spec_kpcn.color:        spec_batched_buffers['color'],
            self.spec_kpcn.grad_x:       spec_batched_buffers['grad_x'],
            self.spec_kpcn.grad_y:       spec_batched_buffers['grad_y'],
            self.spec_kpcn.var_color:    spec_batched_buffers['var_color'],
            self.spec_kpcn.var_features: spec_batched_buffers['var_features'],
            self.albedo:                 batched_albedo,
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration):
        _, loss, loss_summary, gnorm = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm])
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm

    def run_validation(self, sess):
        return sess.run([self.loss, self.diff_kpcn.color, self.out, self.gt_out])

    def save(self, sess, iteration, checkpoint_dir='checkpoints', write_meta_graph=True):
        self.diff_kpcn.save(sess, iteration, os.path.join(checkpoint_dir, 'diff'), write_meta_graph)
        self.spec_kpcn.save(sess, iteration, os.path.join(checkpoint_dir, 'spec'), write_meta_graph)

    def restore(self, sess, restore_path):
        diff_restore_path, spec_restore_path = restore_path
        self.diff_kpcn.restore(sess, diff_restore_path, optimistic=True)
        self.spec_kpcn.restore(sess, spec_restore_path, optimistic=True)