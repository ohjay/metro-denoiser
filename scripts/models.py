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
        self.valid_padding = valid_padding and is_training

        summaries = []  # to merge

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

            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')

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
                            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                '%s/%s/conv2d/kernel' % (self.scope, 'layer%d' % (i + j)))[0]
                            with tf.name_scope('summaries/layer%d' % (i + j)):
                                summaries.append(tf.summary.histogram('conv_kernel_histogram', kernel))
                        elif layer['type'] == 'residual_block':
                            dropout = layer.get('dropout', True)
                            out = self._residual_block(
                                out, self.dropout_keep_prob, self.is_training, kernel_init,
                                layer.get('num_outputs', 100), layer.get('kernel_size', 3), dropout=dropout)
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
                self.out = tf.identity(out, name='out')
            else:
                # KPCN
                self.kernel_size = int(sqrt(layer['num_outputs']))
                out_max =  tf.reduce_max(out, axis=-1, keepdims=True)
                self.out_kernels = tf.nn.softmax(out - out_max, axis=-1)
                _, self.valid_h, self.valid_w, _ = self.out_kernels.get_shape().as_list()
                self.out = tf.identity(
                    self._filter(self.color, self.out_kernels), name='out')  # filtered color buffer

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
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  # for batch norm
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
            self.merged_summaries = tf.summary.merge(summaries)

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
    def _residual_block(_in, dropout_keep_prob, is_training, kernel_init=None,
                        num_outputs=100, kernel_size=3, batchnorm=True, dropout=True):
        with tf.variable_scope('residual_block'):
            out = _in
            for k in range(2):
                out = tf.layers.conv2d(
                    out, filters=num_outputs, kernel_size=kernel_size, strides=1,
                    padding='same', activation=None, kernel_initializer=kernel_init)
                if batchnorm:
                    with tf.variable_scope("batchnorm%d" % k):
                        out = tf.layers.batch_normalization(
                            out, training=is_training, name='batch_normalization')
                out = tf.nn.relu(out)
                if dropout:
                    out = tf.nn.dropout(out, dropout_keep_prob)
        return _in + out

    def run(self, sess, batched_buffers, tensor=None):
        feed_dict = {
            self.color: batched_buffers['color'],
            self.grad_x: batched_buffers['grad_x'],
            self.grad_y: batched_buffers['grad_y'],
            self.var_color: batched_buffers['var_color'],
            self.var_features: batched_buffers['var_features'],
            self.is_training: False,
            self.dropout_keep_prob: 1.0,
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration, dropout_keep_prob):
        feed_dict = {
            self.is_training: True,
            self.dropout_keep_prob: dropout_keep_prob,
        }
        _, loss, loss_summary, gnorm, merged_summaries = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm, self.merged_summaries], feed_dict)
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm, merged_summaries

    def run_validation(self, sess):
        feed_dict = {
            self.is_training: False,
            self.dropout_keep_prob: 1.0,
        }
        return sess.run([self.loss, self.color, self.out, self.gt_out], feed_dict)

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
            if optimistic:
                self._optimistic_restore(sess, restore_path)  # ignore vars not in checkpoint
            else:
                self.saver.restore(sess, restore_path)
            print('[+] `%s` network restored from `%s`.' % (self.scope, restore_path))

class MultiscaleModel(DKPCN):
    """Wrapper around a standard KPCN which combines results from three different scales."""

    def __init__(self, tf_buffers1, buffer_h, buffer_w, layers_config,
                 is_training, learning_rate, summary_dir, scope=None,
                 save_best=False, fp16=False, clip_by_global_norm=False,
                 valid_padding=False, asymmetric_loss=True, sess=None, reuse=False):

        summaries = []
        if scope is None:
            self.sc_scope = 'scale_compositor'
        else:
            self.sc_scope = '/'.join([scope, 'scale_compositor'])

        h1, w1 = MultiscaleModel._get_spatial_dims(tf_buffers1['color'])
        if is_training and (h1 % 4 != 0 or w1 % 4 != 0):
            # make dims % 4 == 0
            tf_buffers1 = self._trim_all(tf_buffers1, h1 % 4, w1 % 4)
            h1, w1 = MultiscaleModel._get_spatial_dims(tf_buffers1['color'])
        self.tf_buffers1 = tf_buffers1

        tf_buffers2 = self._downsample2_all(
            self._trim_all(tf_buffers1, h1 % 2, w1 % 2))
        h2, w2 = MultiscaleModel._get_spatial_dims(tf_buffers2['color'])
        tf_buffers2['var_color']    = tf_buffers2['var_color']    / 4.0
        tf_buffers2['var_features'] = tf_buffers2['var_features'] / 4.0

        tf_buffers4 = self._downsample2_all(
            self._trim_all(tf_buffers2, h2 % 2, w2 % 2))
        h4, w4 = MultiscaleModel._get_spatial_dims(tf_buffers4['color'])
        tf_buffers4['var_color']    = tf_buffers4['var_color']    / 4.0
        tf_buffers4['var_features'] = tf_buffers4['var_features'] / 4.0

        # fine to coarse
        self.kpcn1 = DKPCN(
            tf_buffers1, h1, w1, layers_config, is_training, learning_rate,
            summary_dir, scope, save_best, fp16, clip_by_global_norm,
            valid_padding, asymmetric_loss, sess=None, reuse=tf.AUTO_REUSE)
        self.kpcn2 = DKPCN(
            tf_buffers2, h2, w2, layers_config, is_training, learning_rate,
            summary_dir, scope, save_best, fp16, clip_by_global_norm,
            valid_padding, asymmetric_loss, sess=None, reuse=tf.AUTO_REUSE)
        self.kpcn4 = DKPCN(
            tf_buffers4, h4, w4, layers_config, is_training, learning_rate,
            summary_dir, scope, save_best, fp16, clip_by_global_norm,
            valid_padding, asymmetric_loss, sess=None, reuse=tf.AUTO_REUSE)

        self.denoised1 = tf.stop_gradient(self.kpcn1.out)
        self.denoised2 = tf.stop_gradient(self.kpcn2.out)
        self.denoised4 = tf.stop_gradient(self.kpcn4.out)

        self.out2, self.alpha2 = self._scale_compositor(self.denoised2, self.denoised4)
        self.out, self.alpha1 = self._scale_compositor(self.denoised1, self.out2)  # reuse vars
        self.alpha2_mean = tf.reduce_mean(self.alpha2)
        self.alpha1_mean = tf.reduce_mean(self.alpha1)

        _id = scope or 'x'
        to_log = [
            ('%s/multiscale/color1_in'   % _id, tf_buffers1['color']),
            ('%s/multiscale/color2_in'   % _id, tf_buffers2['color']),
            ('%s/multiscale/color4_in'   % _id, tf_buffers4['color']),
            ('%s/multiscale/color1_out'  % _id, self.denoised1),
            ('%s/multiscale/color2_out'  % _id, self.denoised2),
            ('%s/multiscale/color4_out'  % _id, self.denoised4),
            ('%s/multiscale/pred_alpha2' % _id, self.alpha2),
            ('%s/multiscale/pred_alpha1' % _id, self.alpha1),
        ]
        for name, im in to_log:
            summaries.append(tf.summary.image(name, im, max_outputs=1))
        summaries.append(tf.summary.scalar('%s/multiscale/alpha2_mean' % _id, self.alpha2_mean))
        summaries.append(tf.summary.scalar('%s/multiscale/alpha1_mean' % _id, self.alpha1_mean))

        self.color = tf_buffers1['color']
        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.sc_scope)
        self.saver = tf.train.Saver(self.tvars, max_to_keep=5)

        with tf.variable_scope('/'.join([self.sc_scope, 'opt']), reuse=tf.AUTO_REUSE):
            # loss
            self.gt_out = tf.identity(tf_buffers1['gt_out'], name='gt_out')  # (?, h, w, 3)
            if asymmetric_loss:
                loss = self._asymmetric_smape(self.color, self.out, self.gt_out, slope=2.0)
            else:
                loss = self._smape(self.out, self.gt_out)
            loss = tf.verify_tensor_all_finite(loss, 'NaN or Inf in loss')
            self.loss = tf.identity(loss, name='loss')
            self.loss_summary = tf.summary.scalar('ms_loss', self.loss)
            _loss = self.loss
            if is_training:
                # we know that gt_out will have spatial dims % 4 == 0
                coarse_wt = 0.9
                _loss += coarse_wt * self._smape(self.out2, self._downsample2(self.gt_out))
            alpha_wt = 0.01
            _loss += \
                + alpha_wt * tf.maximum(0.5 - self.alpha2_mean, 0.0) \
                + alpha_wt * tf.maximum(0.5 - self.alpha1_mean, 0.0)  # don't let alpha go to 0

            # optimization
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100000, 0.96)
            if fp16:
                opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)
            else:
                opt = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = opt.compute_gradients(_loss, self.tvars)  # [(grad, var) tuples]
            grads_and_vars = filter(lambda gv: None not in gv, grads_and_vars)
            grads, _vars = zip(*grads_and_vars)
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
        self.merged_summaries = tf.summary.merge(summaries)

    def run(self, sess, batched_buffers, tensor=None):
        feed_dict = {
            self.tf_buffers1['color']: batched_buffers['color'],
            self.tf_buffers1['grad_x']: batched_buffers['grad_x'],
            self.tf_buffers1['grad_y']: batched_buffers['grad_y'],
            self.tf_buffers1['var_color']: batched_buffers['var_color'],
            self.tf_buffers1['var_features']: batched_buffers['var_features'],
            self.kpcn1.is_training: False,
            self.kpcn2.is_training: False,
            self.kpcn4.is_training: False,
            self.kpcn1.dropout_keep_prob: 1.0,
            self.kpcn2.dropout_keep_prob: 1.0,
            self.kpcn4.dropout_keep_prob: 1.0,
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration, dropout_keep_prob):
        feed_dict = {
            self.kpcn1.is_training: True,
            self.kpcn2.is_training: True,
            self.kpcn4.is_training: True,
            self.kpcn1.dropout_keep_prob: dropout_keep_prob,
            self.kpcn2.dropout_keep_prob: dropout_keep_prob,
            self.kpcn4.dropout_keep_prob: dropout_keep_prob,
        }
        _, loss, loss_summary, gnorm, merged_summaries = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm, self.merged_summaries], feed_dict)
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm, merged_summaries

    def run_validation(self, sess):
        feed_dict = {
            self.kpcn1.is_training: False,
            self.kpcn2.is_training: False,
            self.kpcn4.is_training: False,
            self.kpcn1.dropout_keep_prob: 1.0,
            self.kpcn2.dropout_keep_prob: 1.0,
            self.kpcn4.dropout_keep_prob: 1.0,
        }
        return sess.run([self.loss, self.color, self.out, self.gt_out], feed_dict)

    def save(self, sess, iteration, checkpoint_dir='checkpoints', write_meta_graph=True):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        base_filepath = os.path.join(checkpoint_dir, 'ms-var')
        self.saver.save(sess, base_filepath, global_step=iteration, write_meta_graph=write_meta_graph)
        print('[+] Saved current parameters to %s-%d.' % (base_filepath, iteration))

    def restore(self, sess, restore_path, optimistic=False):
        kpcn_rpath, ms_rpath = restore_path
        # restore single-frame KPCN
        self.kpcn1.restore(sess, kpcn_rpath, optimistic=True)  # reuse vars, so should restore all
        # restore multiscale module
        if os.path.isfile(ms_rpath + '.index'):
            if optimistic:
                self._optimistic_restore(sess, ms_rpath)  # ignore vars not in checkpoint
            else:
                self.saver.restore(sess, ms_rpath)
            print('[+] multiscale `%s` network restored from `%s`.' % (self.kpcn1.scope, ms_rpath))

    def _scale_compositor(self, denoised_fine, denoised_coarse):
        h, w = MultiscaleModel._get_spatial_dims(denoised_fine)
        _denoised_fine = self._trim(denoised_fine, h % 2, w % 2)
        UD_fine = self._upsample2(self._downsample2(_denoised_fine))
        U_coarse = self._upsample2(denoised_coarse)
        if h % 2 == 1 or w % 2 == 1:
            UD_fine  = self._pad1(UD_fine,  (h % 2 == 1, w % 2 == 1))
            U_coarse = self._pad1(U_coarse, (h % 2 == 1, w % 2 == 1))

        with tf.variable_scope(self.sc_scope, reuse=tf.AUTO_REUSE):
            alpha = tf.concat((denoised_fine, U_coarse), axis=-1)
            alpha = tf.layers.conv2d(
                alpha, filters=50, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
            alpha = self._residual_block(alpha, None, None, num_outputs=50, batchnorm=False, dropout=False)
            alpha = self._residual_block(alpha, None, None, num_outputs=50, batchnorm=False, dropout=False)
            alpha = tf.layers.conv2d(
                alpha, filters=1, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)
            # blend fine and coarse images
            blend = denoised_fine + alpha * (U_coarse - UD_fine)
            return blend, alpha

    def _downsample2_all(self, buffers):
        return {name: self._downsample2(im) for name, im in buffers.viewitems()}

    @staticmethod
    def _downsample2(im):
        _, _, _, c = im.get_shape().as_list()
        _filter = tf.constant(0.25, shape=[2, 2, 1, 1])
        out = []
        for i in range(c):
            out.append(
                tf.nn.conv2d(im[:, :, :, i:i+1], _filter, strides=[1, 2, 2, 1], padding='SAME'))
        return tf.concat(out, axis=-1)

    @staticmethod
    def _upsample2(im):
        _, h, w, c = im.get_shape().as_list()
        _filter = tf.constant(1.0, shape=[2, 2, 1, 1])
        output_shape = [tf.shape(im)[0], h * 2, w * 2, 1]
        strides = [1, 2, 2, 1]
        out = []
        for i in range(c):
            out.append(
                tf.nn.conv2d_transpose(
                    im[:, :, :, i:i+1], _filter, output_shape, strides, padding='SAME'))
        return tf.concat(out, axis=-1)

    def _trim_all(self, buffers, amt_y, amt_x):
        """Assumes that each image in BUFFERS is of shape (n, h, w, c)."""
        buffers_out = {}
        for k in buffers.viewkeys():
            buffers_out[k] = self._trim(buffers[k], amt_y, amt_x)
        return buffers_out

    @staticmethod
    def _trim(im, amt_y, amt_x):
        _, h, w, _ = im.get_shape().as_list()
        return im[:, :h-amt_y, :w-amt_x, :]

    @staticmethod
    def _pad1(im, which_dims):
        return tf.pad(im, [[0, 0], [0, int(which_dims[0])], [0, int(which_dims[1])], [0, 0]])

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
                 learning_rate, summary_dir, fp16=False, clip_by_global_norm=False):

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

        loss = DKPCN._smape(self.out, self.gt_out)
        loss = tf.verify_tensor_all_finite(loss, 'NaN or Inf in loss')
        self.loss = tf.identity(loss, name='loss')
        self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100000, 0.96)
        if fp16:
            opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)
        else:
            opt = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # for batch norm
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
            self.diff_kpcn.is_training: False,
            self.spec_kpcn.is_training: False,
            self.diff_kpcn.dropout_keep_prob: 1.0,
            self.spec_kpcn.dropout_keep_prob: 1.0,
        }
        if tensor is None:
            tensor = self.out  # tensor to evaluate
        return sess.run(tensor, feed_dict)

    def run_train_step(self, sess, iteration, dropout_keep_prob):
        feed_dict = {
            self.diff_kpcn.is_training: True,
            self.spec_kpcn.is_training: True,
            self.diff_kpcn.dropout_keep_prob: dropout_keep_prob,
            self.spec_kpcn.dropout_keep_prob: dropout_keep_prob,
        }
        _, loss, loss_summary, gnorm = sess.run(
            [self.opt_op, self.loss, self.loss_summary, self.gnorm], feed_dict)
        self.train_writer.add_summary(loss_summary, iteration)
        return loss, gnorm, None

    def run_validation(self, sess):
        feed_dict = {
            self.diff_kpcn.is_training: False,
            self.spec_kpcn.is_training: False,
            self.diff_kpcn.dropout_keep_prob: 1.0,
            self.spec_kpcn.dropout_keep_prob: 1.0,
        }
        return sess.run([self.loss, self.diff_kpcn.color, self.out, self.gt_out], feed_dict)

    def save(self, sess, iteration, checkpoint_dir='checkpoints', write_meta_graph=True):
        self.diff_kpcn.save(sess, iteration, os.path.join(checkpoint_dir, 'diff'), write_meta_graph)
        self.spec_kpcn.save(sess, iteration, os.path.join(checkpoint_dir, 'spec'), write_meta_graph)

    def restore(self, sess, restore_path):
        diff_restore_path, spec_restore_path = restore_path
        self.diff_kpcn.restore(sess, diff_restore_path, optimistic=True)
        self.spec_kpcn.restore(sess, spec_restore_path, optimistic=True)
