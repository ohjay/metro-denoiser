import os
from math import sqrt
import tensorflow as tf

class KPCN(object):
    """
    Kernel-predicting convolutional network,
    as described in Bako et al. 2017.

    The same graph definition is used for both the diffuse and specular networks.
    """
    curr_index = -1

    def __init__(self, buffer_h, buffer_w, layers_config, is_training, learning_rate, summary_dir, scope=None, dtype=tf.float16):

        self.is_training = is_training

        if scope is None:
            KPCN.curr_index += 1
        self.scope = scope or 'KPCN_%d' % KPCN.curr_index
        with tf.variable_scope(self.scope, reuse=False):
            with tf.variable_scope('inputs'):
                # color buffer
                self.color = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 3), name='color')
                # gradients (color, surface normals, albedo, depth)
                self.grad_x = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 10), name='grad_x')
                self.grad_y = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 10), name='grad_y')
                # variance
                self.var_color = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 1), name='var_color')
                self.var_features = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 3), name='var_features')

            out = tf.concat((
                self.color, self.grad_x, self.grad_y, self.var_color, self.var_features), axis=3)

            for i, layer in enumerate(layers_config):
                with tf.variable_scope('layer%d' % i):
                    try:
                        activation = getattr(tf.nn, layer.get('activation', ''))
                    except AttributeError:
                        activation = None
                    if layer['type'] == 'conv2d':
                        out = tf.layers.conv2d(
                            out, layer['num_outputs'], layer['kernel_size'],
                            strides=layer['stride'], padding=layer['padding'], activation=activation, name='conv2d')
                    elif layer['type'] == 'batch_normalization':
                        out = tf.layers.batch_normalization(out, training=self.is_training, name='batch_normalization')
                    else:
                        raise ValueError('unsupported KPCN layer type')

            self.out_kernels = tf.nn.softmax(out, axis=-1)
            self.kernel_size = int(sqrt(layer['num_outputs']))
            self.out = self._filter(self.color, self.out_kernels)  # filtered color buffer

            # loss
            self.gt_out = tf.placeholder(dtype, shape=(None, buffer_h, buffer_w, 3), name='gt_out')
            self.loss = tf.reduce_mean(tf.abs(self.out - self.gt_out), name='l1_loss')
            self.loss_summary = tf.summary.scalar('loss', self.loss)

            # optimization
            opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4)  # https://stackoverflow.com/a/42077538
            grads_and_vars = opt.compute_gradients(self.loss)
            self.opt_op = opt.apply_gradients(grads_and_vars)

            # logging
            self.train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'))

    def _filter(self, color, kernels):

        # `color`   : (?, h, w, 3)
        # `kernels` : (?, h, w, kernel_size * kernel_size)

        # zero-pad color buffer
        kernel_radius = self.kernel_size // 2
        color = tf.pad(color, [
            [0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])

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

    def run(self, sess, batched_buffers):
        feed_dict = {
            self.color: batched_buffers['color'],
            self.grad_x: batched_buffers['grad_x'],
            self.grad_y: batched_buffers['grad_y'],
            self.var_color: batched_buffers['var_color'],
            self.var_features: batched_buffers['var_features'],
        }
        return sess.run(self.out, feed_dict)

    def run_train_step(self, sess, batched_buffers, gt_out, iteration):
        feed_dict = {
            self.color: batched_buffers['color'],
            self.grad_x: batched_buffers['grad_x'],
            self.grad_y: batched_buffers['grad_y'],
            self.var_color: batched_buffers['var_color'],
            self.var_features: batched_buffers['var_features'],
            self.gt_out: gt_out,
        }
        _, loss, loss_summary = sess.run([
            self.opt_op, self.loss, self.loss_summary], feed_dict)
        self.train_writer.add_summary(loss_summary, iteration)
        return loss

    def save(self, sess, iteration, checkpoint_dir='checkpoints', write_meta_graph=True):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        base_filepath = os.path.join(checkpoint_dir, 'var')
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(sess, base_filepath, global_step=iteration, write_meta_graph=write_meta_graph)
        print('[+] Saved current parameters to %s-%d.' % (base_filepath, iteration))

    def restore(self, sess, iteration, checkpoint_dir='checkpoints'):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(sess, os.path.join(checkpoint_dir, 'var-%d' % iteration))
        print('[+] `%s` KPCN restored to iteration %d (checkpoint_dir=%s).' % (self.scope, iteration, checkpoint_dir))
