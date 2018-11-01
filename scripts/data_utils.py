import os
import cv2
import time
import Imath
import random
import OpenEXR
import numpy as np
from math import sqrt
import tensorflow as tf
from scipy.misc import imsave
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# ===============================================
# EXR I/O
# ===============================================

def read_exr(filepath, fp16=True):
    """
    Reads EXR file as dictionary of NumPy arrays.
    `read_exr` and `write_exr` are inverse operations.
    """
    exr = OpenEXR.InputFile(filepath)
    header = exr.header()
    wind = header['dataWindow']
    im_shape = (wind.max.y - wind.min.y + 1, wind.max.x - wind.min.x + 1)

    if fp16:
        pt_dtype = Imath.PixelType.HALF
        np_dtype = np.float16
    else:
        pt_dtype = Imath.PixelType.FLOAT
        np_dtype = np.float32
    pt = Imath.PixelType(pt_dtype)

    buffers = {}
    for c in header['channels'].keys():
        c_components = c.split('.')
        curr_dict = buffers
        for component in c_components[:-1]:
            if component not in curr_dict:
                curr_dict[component] = {}
            curr_dict = curr_dict[component]
        component = c_components[-1]
        curr_dict[component] = np.fromstring(exr.channel(c, pt), dtype=np_dtype)
        curr_dict[component].shape = im_shape

    return buffers
    # RGB image: np.stack([buffers['R'], buffers['G'], buffers['B']], axis=-1)

def write_exr(buffers, filepath):
    """
    Write dictionary of NumPy arrays to EXR file.
    `read_exr` and `write_exr` are inverse operations.
    """
    if buffers['R'].dtype == np.float16:
        pt_dtype = Imath.PixelType.HALF
    else:
        pt_dtype = Imath.PixelType.FLOAT

    channel = Imath.Channel(Imath.PixelType(pt_dtype))
    header = OpenEXR.Header(*buffers['R'].shape[::-1])
    header['channels'] = {}
    pixels = {}

    def assign_channels(curr_dict, c=''):
        for component, data in curr_dict.items():
            c_components = '.'.join(filter(None, [c, component]))
            if type(data) == dict:
                assign_channels(data, c_components)
            else:
                header['channels'][c_components] = channel
                pixels[c_components] = data
    assign_channels(buffers)

    exr = OpenEXR.OutputFile(filepath, header)
    exr.writePixels(
        dict([(c, data.tostring()) for c, data in pixels.items()]))
    exr.close()

def stack_channels(buffers):
    """
    Stack multi-channel data.
    Currently, BUFFERS is a nested dictionary as produced by `read_exr`.
    Assumption: no more than one layer of nesting.

    If, e.g., BUFFERS = {'diffuse': {'R': arrR, 'G': arrG, 'B': arrB}},
    this function will return {'diffuse': np.stack([arrR, arrG, arrB], axis=-1)}.
    Note: if channels are named 'R', 'G', and 'B', will order as RGB.

    The return value is a dictionary of the form {'name': [h, w, c]-array}.

    (`read_exr` -> `stack_channels`) and
    (`split_channels` -> `write_exr`) are inverse operations.
    """
    buffers_out = {}
    for c, data in buffers.items():
        if type(data) == dict:
            data_channels = []
            if 'R' in data:
                data_channels.append(data['R'])
            if 'G' in data:
                data_channels.append(data['G'])
            if 'B' in data:
                data_channels.append(data['B'])
            for component in data:
                if component not in {'R', 'G', 'B'}:
                    data_channels.append(data[component])
            buffers_out[c] = np.stack(data_channels, axis=-1)
        else:
            buffers_out[c] = np.expand_dims(data, -1)
    return buffers_out

def split_channels(buffers):
    """
    Split multi-channel data.
    Currently, BUFFERS is a non-nested dictionary.

    Assumption: 1 channel, 'R'/'G'/'B'     -> (same).
    Assumption: 1 channel, non 'R'/'G'/'B' -> Z.
    Assumption: 3 channels                 -> R, G, B.

    (`read_exr` -> `stack_channels`) and
    (`split_channels` -> `write_exr`) are inverse operations.
    """
    buffers_out = {}
    for c, data in buffers.items():
        if c in {'R', 'G', 'B'}:
            buffers_out[c] = data[:, :, 0]
        elif data.shape[-1] == 1:
            buffers_out[c] = {
                'Z': data[:, :, 0]}
        else:
            buffers_out[c] = {
                'R': data[:, :, 0],
                'G': data[:, :, 1],
                'B': data[:, :, 2]}
    return buffers_out

# ===============================================
# TFRECORDS I/O
# ===============================================

N_CHANNELS = {
    'diffuse':          3,
    'diffuseVariance':  1,
    'specular':         3,
    'specularVariance': 1,
    'albedo':           3,
    'albedoVariance':   1,
    'normal':           3,
    'normalVariance':   1,
    'depth':            1,
    'depthVariance':    1,
    'gt_diffuse':       3,
    'gt_specular':      3,
    'gt_albedo':        3,
}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(tfrecord_filepath, input_exr_files, gt_exr_files,
                    patches_per_im, patch_size, fp16, shuffle=False, debug_dir='',
                    save_debug_ims_every=1, color_var_weight=1.0, normal_var_weight=1.0):
    """Export PATCHES_PER_IM examples for each EXR file.
    Accepts two lists of EXR filepaths with corresponding orderings.

    Handles patch sampling but not preprocessing.
    """
    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    with tf.python_io.TFRecordWriter(tfrecord_filepath) as writer:
        all_examples = []
        all_examples_size_limit = patches_per_im * 50  # essentially, shuffle buffer lim = 50 ims

        def shuffle_and_write():
            random.shuffle(all_examples)
            for example in all_examples:
                writer.write(example.SerializeToString())
            print('[+] Wrote %d examples to TFRecord file `%s`.' % (len(all_examples), tfrecord_filepath))

        i = 0
        start_time = time.time()
        for input_filepath, gt_filepath in zip(input_exr_files, gt_exr_files):
            # for filename str printing
            input_dirname = os.path.dirname(input_filepath)
            input_basename = os.path.basename(input_filepath)
            input_id = os.path.join(os.path.basename(input_dirname), input_basename)

            input_buffers = read_exr(input_filepath, fp16=fp16)
            input_buffers = stack_channels(input_buffers)
            try:
                _input_id  = input_basename[:input_basename.rfind('.')]
                _debug_dir = debug_dir if i % save_debug_ims_every == 0 else ''
                patch_indices = sample_patches(
                    input_buffers, patches_per_im, patch_size, patch_size, _debug_dir, _input_id,
                    color_var_weight=color_var_weight, normal_var_weight=normal_var_weight)
            except ValueError as e:
                print('[-] Invalid value during %s sampling. (%s)' % (input_id, str(e)))
                continue

            gt_buffers = read_exr(gt_filepath, fp16=fp16)
            gt_buffers = stack_channels(gt_buffers)

            # One example per patch
            r = patch_size // 2
            for y, x in patch_indices:
                feature = {
                    'diffuse':          input_buffers['diffuse'][y-r:y+r+1, x-r:x+r+1, :],
                    'diffuseVariance':  input_buffers['diffuseVariance'][y-r:y+r+1, x-r:x+r+1, :],
                    'specular':         input_buffers['specular'][y-r:y+r+1, x-r:x+r+1, :],
                    'specularVariance': input_buffers['specularVariance'][y-r:y+r+1, x-r:x+r+1, :],
                    'albedo':           input_buffers['albedo'][y-r:y+r+1, x-r:x+r+1, :],
                    'albedoVariance':   input_buffers['albedoVariance'][y-r:y+r+1, x-r:x+r+1, :],
                    'normal':           input_buffers['normal'][y-r:y+r+1, x-r:x+r+1, :],
                    'normalVariance':   input_buffers['normalVariance'][y-r:y+r+1, x-r:x+r+1, :],
                    'depth':            input_buffers['depth'][y-r:y+r+1, x-r:x+r+1, :],
                    'depthVariance':    input_buffers['depthVariance'][y-r:y+r+1, x-r:x+r+1, :],
                    'gt_diffuse':       gt_buffers['diffuse'][y-r:y+r+1, x-r:x+r+1, :],
                    'gt_specular':      gt_buffers['specular'][y-r:y+r+1, x-r:x+r+1, :],
                    'gt_albedo':        gt_buffers['albedo'][y-r:y+r+1, x-r:x+r+1, :],
                }
                for c, data in feature.items():
                    if fp16:
                        feature[c] = _bytes_feature(data.tobytes())
                    else:
                        feature[c] = _bytes_feature(tf.compat.as_bytes(data.tostring()))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                if shuffle:
                    all_examples.append(example)
                else:
                    writer.write(example.SerializeToString())
            print('[o] Collected %d examples for %s (%d/%d).'
                % (len(patch_indices), input_id, i + 1, len(input_exr_files)))
            if (i + 1) % 10 == 0:
                print('--- TIME ELAPSED: %s' % format_seconds(time.time() - start_time))

            if shuffle and len(all_examples) >= all_examples_size_limit:
                shuffle_and_write()
                del all_examples[:]

            i += 1

        # (final) shuffle
        if shuffle:
            if len(all_examples) > 0:
                shuffle_and_write()
        else:
            print('[+] Wrote examples to TFRecord file `%s`.' % tfrecord_filepath)

def make_decode(is_diffuse, tf_dtype, buffer_h, buffer_w, eps, clip_ims):

    def decode(serialized_example):
        """De-serialize and preprocess TFRecord example."""
        _features = {
            'albedo':         tf.FixedLenFeature([], tf.string),
            'albedoVariance': tf.FixedLenFeature([], tf.string),
            'normal':         tf.FixedLenFeature([], tf.string),
            'normalVariance': tf.FixedLenFeature([], tf.string),
            'depth':          tf.FixedLenFeature([], tf.string),
            'depthVariance':  tf.FixedLenFeature([], tf.string),
        }
        if is_diffuse:
            _features['diffuse'] = tf.FixedLenFeature([], tf.string)
            _features['diffuseVariance'] = tf.FixedLenFeature([], tf.string)
            _features['gt_diffuse'] = tf.FixedLenFeature([], tf.string)
            _features['gt_albedo'] = tf.FixedLenFeature([], tf.string)
        else:
            _features['specular'] = tf.FixedLenFeature([], tf.string)
            _features['specularVariance'] = tf.FixedLenFeature([], tf.string)
            _features['gt_specular'] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(serialized_example, _features)
        p = {}  # "p" for "parsed"
        for name in _features.keys():
            p[name] = tf.decode_raw(features[name], tf_dtype)
            p[name] = tf.reshape(p[name], [buffer_h, buffer_w, N_CHANNELS[name]])

        # clipping
        if clip_ims:
            if is_diffuse:
                p['diffuse']     = tf.clip_by_value(p['diffuse'], 0.0, 1.0)
                p['gt_diffuse']  = tf.clip_by_value(p['gt_diffuse'], 0.0, 1.0)
            else:
                p['specular']    = tf.clip_by_value(p['specular'], 0.0, 1.0)
                p['gt_specular'] = tf.clip_by_value(p['gt_specular'], 0.0, 1.0)

        # preprocess
        if is_diffuse:
            p['diffuse'] = tf_preprocess_diffuse(p['diffuse'], p['albedo'], eps)
            p['gt_diffuse'] = tf_preprocess_diffuse(p['gt_diffuse'], p['gt_albedo'], eps)
            p['diffuseVariance'] = tf_preprocess_diffuse_variance(p['diffuseVariance'], p['albedo'], eps)
        else:
            p['specular'] = tf_preprocess_specular(p['specular'])
            p['gt_specular'] = tf_preprocess_specular(p['gt_specular'])
            p['specularVariance'] = tf_preprocess_specular_variance(p['specular'], p['specularVariance'])
        p['depth'], p['depthVariance'] = tf_preprocess_depth(p['depth'], p['depthVariance'])

        variance_features = tf.concat([
            p['normalVariance'],
            p['albedoVariance'],
            p['depthVariance']], axis=-1)

        return (
            p['diffuse'] if is_diffuse else p['specular'],
            p['normal'],
            p['albedo'],
            p['depth'],
            p['diffuseVariance'] if is_diffuse else p['specularVariance'],
            variance_features,
            p['gt_diffuse'] if is_diffuse else p['gt_specular'],
        )  # i.e. always return 7 tensors

    return decode

# ===============================================
# SAMPLING
# ===============================================

def to_uint8(im):
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)

def approximately_black(im, mean_threshold=1, std_threshold=1, count_threshold=3):
    """Return True if image is approximately black."""
    im = to_uint8(im)
    return np.mean(im) <= mean_threshold \
        and np.std(im) <= std_threshold \
        and np.count_nonzero(im) <= count_threshold

def compute_rel_luminance(im):
    """Compute relative luminance from RGB image as per goo.gl/YddtR4."""
    return 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]

def compute_variance(im, window_h, window_w):
    """Compute variance image.
    Source: https://stackoverflow.com/a/36266187.
    """
    kernel_size     = (window_h, window_w)
    window_mean     = cv2.boxFilter(im,      -1, kernel_size, borderType=cv2.BORDER_REFLECT)
    window_sqr_mean = cv2.boxFilter(im ** 2, -1, kernel_size, borderType=cv2.BORDER_REFLECT)
    return window_sqr_mean - window_mean ** 2

def compute_processed_variance(im, window_h, window_w):
    """Compute smoothed, single-channel variance image."""
    variance = compute_variance(im, window_h, window_w)
    variance = compute_rel_luminance(variance)
    variance = gaussian_filter_wrapper(variance, sigma=1.0)
    return np.clip(variance / np.amax(variance), 0.0, 1.0)

def invalid_complain(data):
    """Raise an error if there are invalid values in the given NumPy array."""
    if np.count_nonzero(data) == 0:
        raise ValueError('data all zero')
    if np.count_nonzero(np.isnan(data)) > 0:
        raise ValueError('invalid value: NaN')
    if np.count_nonzero(np.isinf(data)) > 0:
        raise ValueError('invalid value: inf')

def gaussian_filter_wrapper(signal, sigma=1.0):
    """Perform Gaussian blurring, allowing for fp16 dtype."""
    fp16 = signal.dtype == np.float16
    if fp16:
        signal = signal.astype(np.float32)
    signal = gaussian_filter(signal, sigma=1.0)
    if fp16:
        signal = signal.astype(np.float16)
    return signal

def sample_patches(buffers, num_patches, patch_h, patch_w, debug_dir, input_id,
                   color_var_weight=1.0, normal_var_weight=1.0):
    """
    Sample NUM_PATCHES (y, x) indices for (PATCH_H, PATCH_W)-sized patches from the given frame.
    As per Bako et al., we will find candidate patches and prune based on color/normal variance.
    Operates in EXR buffer space (as opposed to tensor space).

    We will sample min(NUM_PATCHES, number of nonzero patches), rather than NUM_PATCHES exactly.
    """
    h, w = buffers['normal'].shape[:2]
    y_range = (patch_h // 2, h - patch_h // 2)  # [)
    x_range = (patch_w // 2, w - patch_w // 2)  # [)

    color = clip_and_gamma_correct(
        np.concatenate([buffers['R'], buffers['G'], buffers['B']], axis=-1))

    # 2D PDF ---------------------------------------------------------------

    c_var = compute_processed_variance(color, patch_h, patch_w)
    n_var = compute_processed_variance(buffers['normal'], patch_h, patch_w)

    pdf = color_var_weight * c_var + normal_var_weight * n_var
    if color_var_weight > 0:
        pdf *= (c_var > 0)  # no zero-variance patches

    # Set out-of-bounds regions to zero
    _template = np.zeros_like(pdf)
    _template[y_range[0]:y_range[1], x_range[0]:x_range[1]] \
        = pdf[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    pdf = _template

    invalid_complain(pdf)
    pdf /= np.sum(pdf)  # normalize to [0, 1], sum to 1

    # SAMPLING -------------------------------------------------------------

    num_samples = min(num_patches, np.count_nonzero(pdf))
    samples = np.random.choice(pdf.size, num_samples, replace=False, p=pdf.flatten())
    patch_indices = np.unravel_index(samples, pdf.shape)
    patch_indices = np.stack(patch_indices, axis=1)  # (num_samples, 2)
    print('[o] Sampled %d patch indices.' % num_samples)

    # DEBUG ----------------------------------------------------------------

    if debug_dir:
        # tile patches into debug image
        rh = patch_h // 2
        rw = patch_w // 2
        rowlen = int(sqrt(num_samples)) + 1
        nrows = (num_samples - 1) // rowlen + 1  # works because I know num_samples > 0
        ncols = num_samples % rowlen if num_samples < rowlen else rowlen
        patches = np.zeros((nrows * patch_h, ncols * patch_w, 3))
        patches_overlay = color * 0.1  # credit to Bako et al. for this visualization idea
        for i, (y, x) in enumerate(patch_indices):
            py = (i // rowlen) * patch_h
            px = (i % rowlen)  * patch_w
            patches[py:py+patch_h, px:px+patch_w] = color[y-rh:y+rh+1, x-rw:x+rw+1, :]
            patches_overlay[y-rh:y+rh+1, x-rw:x+rw+1, :] = color[y-rh:y+rh+1, x-rw:x+rw+1, :]

        qsize = (w // 2, h // 2)  # w, h ordering
        _c_var = cv2.resize(c_var, qsize)
        _n_var = cv2.resize(n_var, qsize)
        _pdf   = cv2.resize(pdf,   qsize)

        imsave(os.path.join(debug_dir, input_id + '_00_color_var.jpg'),       _c_var)
        imsave(os.path.join(debug_dir, input_id + '_01_normal_var.jpg'),      _n_var)
        imsave(os.path.join(debug_dir, input_id + '_02_pdf.jpg'),             _pdf)
        imsave(os.path.join(debug_dir, input_id + '_03_patches.jpg'),         patches)
        imsave(os.path.join(debug_dir, input_id + '_04_patches_overlay.jpg'), patches_overlay)

    return patch_indices

# ===============================================
# PRE/POST-PROCESSING (NUMPY)
# ===============================================

def clip_and_gamma_correct(im):
    assert im.dtype != np.uint8
    return np.clip(im, 0.0, 1.0) ** (1.0 / 2.2)

def preprocess_diffuse(buffers, eps):
    """Factor out albedo. Destructive."""
    buffers['diffuse'] /= buffers['albedo'] + eps
    mean_albedo = np.mean(buffers['albedo'], axis=-1)
    mean_albedo = np.expand_dims(mean_albedo, -1)
    buffers['diffuseVariance'] /= np.square(mean_albedo + eps)

def preprocess_specular(buffers):
    """Apply logarithmic transform. Destructive."""
    buffers['specular'] = np.log(buffers['specular'] + 1.0)
    mean_specular = np.mean(buffers['specular'], axis=-1)
    mean_specular = np.expand_dims(mean_specular, -1)
    buffers['specularVariance'] /= np.square(mean_specular) + 1e-5

def preprocess_depth(buffers):
    """Scale depth to range [0, 1]. Destructive."""
    _min = np.min(buffers['depth'])
    _range = np.max(buffers['depth']) - _min
    buffers['depth'] = (buffers['depth'] - _min) / _range
    buffers['depthVariance'] /= _range ** 2

def compute_buffer_gradients(buffers):
    """
    Return horizontal and vertical gradients for
    diffuse buffer (3 channels), specular buffer (3 channels),
    normal buffer (3 channels), albedo buffer (3 channels), and depth buffer (1 channel).

    Note: cannot use `np.gradient` because it uses central differences, whereas
    `tf.image.image_gradients`, this function's counterpart, uses forward differences.
    """
    def _image_gradients(_buffer):
        grad_y = np.zeros_like(_buffer)
        grad_y[:-1, :, :] = _buffer[1:, :, :] - _buffer[:-1, :, :]

        grad_x = np.zeros_like(_buffer)
        grad_x[:, :-1, :] = _buffer[:, 1:, :] - _buffer[:, :-1, :]

        return grad_y, grad_x

    diffuse_grad_y,  diffuse_grad_x  = _image_gradients(buffers['diffuse'])
    specular_grad_y, specular_grad_x = _image_gradients(buffers['specular'])
    normal_grad_y,   normal_grad_x   = _image_gradients(buffers['normal'])
    albedo_grad_y,   albedo_grad_x   = _image_gradients(buffers['albedo'])
    depth_grad_y,    depth_grad_x    = _image_gradients(buffers['depth'])

    grad_y = {
        'diffuse': diffuse_grad_y,
        'specular': specular_grad_y,
        'normal': normal_grad_y,
        'albedo': albedo_grad_y,
        'depth': depth_grad_y,
    }
    grad_x = {
        'diffuse': diffuse_grad_x,
        'specular': specular_grad_x,
        'normal': normal_grad_x,
        'albedo': albedo_grad_x,
        'depth': depth_grad_x,
    }
    return grad_y, grad_x

def make_network_inputs(buffers):
    """Takes preprocessed buffers (for a single image, i.e. shapes are HWC)
    and constructs both diffuse and specular KPCN inputs.

    This function does not split the image into patches.
    If such behavior is desired, it can be applied after this function.
    """
    var_features = np.concatenate(
        [buffers['normalVariance'], buffers['albedoVariance'], buffers['depthVariance']], axis=-1)

    grad_y, grad_x = compute_buffer_gradients(buffers)
    grad_y_features = np.concatenate(
        [grad_y['normal'], grad_y['albedo'], grad_y['depth']], axis=-1)
    grad_x_features = np.concatenate(
        [grad_x['normal'], grad_x['albedo'], grad_x['depth']], axis=-1)

    diff_in = {
        'color': buffers['diffuse'],
        'grad_x': np.concatenate([grad_x['diffuse'], grad_x_features], axis=-1),
        'grad_y': np.concatenate([grad_y['diffuse'], grad_y_features], axis=-1),
        'var_color': buffers['diffuseVariance'],
        'var_features': var_features,
    }
    spec_in = {
        'color': buffers['specular'],
        'grad_x': np.concatenate([grad_x['specular'], grad_x_features], axis=-1),
        'grad_y': np.concatenate([grad_y['specular'], grad_y_features], axis=-1),
        'var_color': buffers['specularVariance'],
        'var_features': var_features,
    }

    # add batch dim
    for c, data in diff_in.items():
        diff_in[c] = np.expand_dims(data, 0)
    for c, data in spec_in.items():
        spec_in[c] = np.expand_dims(data, 0)

    return diff_in, spec_in

def postprocess_diffuse(out_diffuse, albedo, eps):
    """Multiply back albedo."""
    return out_diffuse * (albedo + eps)

def postprocess_specular(out_specular):
    """Apply exponential transform."""
    return np.exp(out_specular) - 1.0

# ===============================================
# PRE/POST-PROCESSING (TENSORFLOW)
# ===============================================

def tf_preprocess_diffuse(diffuse, albedo, eps):
    return tf.divide(diffuse, albedo + eps)

def tf_preprocess_diffuse_variance(diffuse_variance, albedo, eps):
    mean_albedo = tf.reduce_mean(albedo, axis=-1)
    mean_albedo = tf.expand_dims(mean_albedo, -1)
    return tf.divide(diffuse_variance, tf.square(mean_albedo + eps))

def tf_preprocess_specular(specular):
    return tf.log(specular + 1.0)

def tf_preprocess_specular_variance(specular, specular_variance):
    mean_specular = tf.reduce_mean(specular, axis=-1)
    mean_specular = tf.expand_dims(mean_specular, -1)
    return tf.divide(specular_variance, tf.square(mean_specular) + 1e-5)

def tf_preprocess_depth(depth, depth_variance):
    _min = tf.reduce_min(depth)
    _range = tf.reduce_max(depth) - _min
    return tf.divide(depth - _min, _range), tf.divide(depth_variance, tf.square(_range))

def tf_postprocess_diffuse(out_diffuse, albedo, eps):
    return tf.multiply(out_diffuse, albedo + eps)

def tf_postprocess_specular(out_specular):
    return tf.exp(out_specular) - 1.0

# ===============================================
# VISUALIZATION
# ===============================================

def show_multiple(*ims, **kwargs):
    """Plot multiple images in one grid.
    By default, the maximum row length is three.

    Assumes that each image in IMS is either
    an [h, w, c]-array or an [n, h, w, c]-array.
    """
    row_max = kwargs.get('row_max', 3)
    batch_max = kwargs.get('batch_max', 5)  # viz at most 5 examples per batch
    block_on_viz = kwargs.get('block_on_viz', False)

    if not block_on_viz:
        plt.ion()
        plt.show()

    # Determine number of rows and columns
    assert len(ims) > 0
    nrows = (len(ims) - 1) // row_max + 1
    ncols = len(ims) % row_max if len(ims) < row_max else row_max
    base_nrows = nrows
    if len(ims[0].shape) == 4:
        nrows *= min(len(ims[0].shape), batch_max)

    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    for j, im in enumerate(ims):
        _im = im
        _nd = len(_im.shape)
        cmap = None
        if _im.dtype == np.float16:
            _im = _im.astype(np.float32)
        if _im.shape[-1] == 1 or _nd == 2:
            cmap = plt.gray()
        if _im.shape[-1] == 1:
            # single-channel (grayscale) image
            _im = np.squeeze(_im, axis=-1)
        if _nd == 4:
            for k in range(min(_nd, batch_max)):
                row = base_nrows * k + j // row_max
                ax[row, j % row_max].imshow(_im[k], cmap=cmap)
                ax[row, j % row_max].set_axis_off()
        else:
            ax[j // row_max, j % row_max].imshow(_im, cmap=cmap)
            ax[j // row_max, j % row_max].set_axis_off()

    if block_on_viz:
        plt.show()
    else:
        try:
            plt.pause(10)
            plt.close()
        except:
            pass  # in case user closes window herself

# ===============================================
# FILESYSTEM I/O
# ===============================================

def get_run_dir(enclosing_dir):
    run_id = 0
    while os.path.isdir(os.path.join(enclosing_dir, 'run_%04d' % run_id)):
        run_id += 1
    return os.path.join(enclosing_dir, 'run_%04d' % run_id)

# ===============================================
# GENERAL
# ===============================================

DAY_FORMAT_STR  = '{d} days, {h} hours, {m} minutes, {s} seconds'
HOUR_FORMAT_STR = '{h} hours, {m} minutes, {s} seconds'
MIN_FORMAT_STR  = '{m} minutes, {s} seconds'
SEC_FORMAT_STR  = '{s} seconds'

def format_seconds(s):
    """Converts S, a float representing some number of seconds,
    into a string detailing DAYS, HOURS, MINUTES, and SECONDS.
    """
    s = int(s)
    seconds = s % 60
    minutes = s // 60
    hours   = s // 3600
    days    = s // 86400

    if days > 0:
        format_str = DAY_FORMAT_STR
    elif hours > 0:
        format_str = HOUR_FORMAT_STR
    elif minutes > 0:
        format_str = MIN_FORMAT_STR
    else:
        format_str = SEC_FORMAT_STR

    return format_str.format(d=days, h=hours, m=minutes, s=seconds)
