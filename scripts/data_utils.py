import os
import re
import cv2
import time
import Imath
import pyexr
import pickle
import random
import OpenEXR
import numpy as np
import tensorflow as tf
from math import sqrt, ceil
from scipy.misc import imsave
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage.measure import compare_ssim
from scipy.ndimage.filters import gaussian_filter

MAX_DEPTH = 20.0  # empirical

# ===============================================
# EXR I/O
# ===============================================

def read_exr(filepath, fp16=True):
    """
    Reads EXR file as dictionary of NumPy arrays.
    The dictionary will be of the form {'name': [h, w, c]-array}.
    """
    if fp16:
        buffers = pyexr.read_all(filepath, precision=pyexr.HALF)
    else:
        buffers = pyexr.read_all(filepath)
    return {c: np.nan_to_num(data) for c, data in buffers.items()}

def write_exr(buffers, filepath):
    """
    Write dictionary of NumPy arrays to EXR file.
    If BUFFERS is a NumPy array with 3 channels, will interpret as RGB and proceed.
    """
    if type(buffers) == np.ndarray and buffers.shape[-1] == 3:
        # convert into dictionary
        assert len(buffers.shape) == 3
        buffers = {'R': buffers[:, :, 0], 'G': buffers[:, :, 1], 'B': buffers[:, :, 2]}

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

def write_tfrecords(tfrecord_filepath, input_exr_files, gt_exr_files, patches_per_im,
                    patch_size, fp16, shuffle=False, debug_dir='', save_debug_ims_every=1,
                    color_var_weight=1.0, normal_var_weight=1.0, file_example_limit=1e5,
                    error_maps=None, indiv_spp=-1, save_integrated_patches=False):
    """Export PATCHES_PER_IM examples for each EXR file.
    Accepts two lists of EXR filepaths with corresponding orderings.

    Handles patch sampling but not preprocessing.
    """
    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    file_idx = 0
    example_buffer = []
    which_patch_indices = {}

    def write_example_buffer():
        if shuffle:
            random.shuffle(example_buffer)
        curr_filepath = tfrecord_filepath % file_idx
        num_to_write  = min(file_example_limit, len(example_buffer))
        with tf.python_io.TFRecordWriter(curr_filepath) as writer:
            for example in example_buffer[:num_to_write]:
                writer.write(example.SerializeToString())
        print('[+] Wrote %d examples to TFRecord file `%s`.' % (num_to_write, curr_filepath))
        del example_buffer[:num_to_write]
        return file_idx + 1  # return next file index

    if indiv_spp > 0:
        i = 0
        start_time = time.time()
        for input_filepaths, gt_filepath in zip(input_exr_files, gt_exr_files):
            # for filename str printing and dictionary saving
            scene_id = os.path.basename(os.path.dirname(os.path.dirname(input_filepaths[0])))
            permu_id = os.path.basename(os.path.dirname(input_filepaths[0]))
            input_id = os.path.join(scene_id, permu_id)

            input_buffers = []
            for k, input_filepath in enumerate(input_filepaths):
                if k % 10 == 0:
                    _if_dir = os.path.basename(os.path.dirname(input_filepath))
                    print('[o] Reading `%s`...' % os.path.join(_if_dir, os.path.basename(input_filepath)))
                input_buffers.append(read_exr(input_filepath, fp16=fp16))

            gt_buffers = read_exr(gt_filepath, fp16=fp16)

            print('[o] Sampling patches...')
            try:
                # Sample based on ground truth buffers
                patch_indices = sample_patches(
                    gt_buffers, patches_per_im, patch_size, patch_size, '', '',
                    color_var_weight=color_var_weight, normal_var_weight=normal_var_weight, pdf=None)
            except ValueError as e:
                print('[-] Invalid value during %s sampling. (%s)' % (input_id, str(e)))
                continue
            which_patch_indices[input_id] = patch_indices

            r = patch_size // 2
            for y, x in patch_indices:
                feature_names = [
                    'diffuse',
                    'diffuseVariance',
                    'specular',
                    'specularVariance',
                    'albedo',
                    'albedoVariance',
                    'normal',
                    'normalVariance',
                    'depth',
                    'depthVariance',
                ]
                feature = {}
                for fname in feature_names:
                    feature[fname] = []
                    for ibuffer in input_buffers:
                        feature[fname].append(ibuffer[fname][y-r:y+r+1, x-r:x+r+1, :])
                    if save_integrated_patches:
                        feature[fname] = np.mean(feature[fname], axis=0)  # all will be (ph, pw, 3)
                    else:
                        feature[fname] = np.array(feature[fname])  # all will be (depth, ph, pw, 3)

                feature['gt_diffuse'] = gt_buffers['diffuse'][y-r:y+r+1, x-r:x+r+1, :]
                feature['gt_specular'] = gt_buffers['specular'][y-r:y+r+1, x-r:x+r+1, :]
                feature['gt_albedo'] = gt_buffers['albedo'][y-r:y+r+1, x-r:x+r+1, :]

                for c, data in feature.items():
                    if fp16:
                        feature[c] = _bytes_feature(data.tobytes())
                    else:
                        feature[c] = _bytes_feature(tf.compat.as_bytes(data.tostring()))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                example_buffer.append(example)
            print('[o] Collected %d examples for %s (%d/%d).'
                  % (len(patch_indices), input_id, i + 1, len(input_exr_files)))
            if (i + 1) % 10 == 0:
                print('(time elapsed: %s)' % format_seconds(time.time() - start_time))

            while len(example_buffer) >= file_example_limit:
                file_idx = write_example_buffer()

            i += 1
    else:
        i = 0
        start_time = time.time()
        for input_filepath, gt_filepath in zip(input_exr_files, gt_exr_files):
            # for filename str printing and dictionary saving
            input_dirname = os.path.dirname(input_filepath)
            input_basename = os.path.basename(input_filepath)
            scene_id = os.path.basename(input_dirname)
            permu_id = input_basename[:input_basename.rfind('-')]
            input_id = os.path.join(scene_id, permu_id)

            sampling_pdf = None
            if error_maps is not None:
                # use error map for sampling
                sampling_pdf = generate_sampling_map(error_maps[input_filepath], patch_size)

            input_buffers = read_exr(input_filepath, fp16=fp16)
            try:
                _input_id  = input_basename[:input_basename.rfind('.')]
                _debug_dir = debug_dir if i % save_debug_ims_every == 0 else ''
                # TODO: sample based on ground truth buffers?
                patch_indices = sample_patches(
                    input_buffers, patches_per_im, patch_size, patch_size, _debug_dir, _input_id,
                    color_var_weight=color_var_weight, normal_var_weight=normal_var_weight, pdf=sampling_pdf)
            except ValueError as e:
                print('[-] Invalid value during %s sampling. (%s)' % (input_id, str(e)))
                continue
            which_patch_indices[os.path.join(scene_id, permu_id)] = patch_indices

            gt_buffers = read_exr(gt_filepath, fp16=fp16)

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
                example_buffer.append(example)
            print('[o] Collected %d examples for %s (%d/%d).'
                % (len(patch_indices), input_id, i + 1, len(input_exr_files)))
            if (i + 1) % 10 == 0:
                print('(time elapsed: %s)' % format_seconds(time.time() - start_time))

            while len(example_buffer) >= file_example_limit:
                file_idx = write_example_buffer()

            i += 1

    # (final) write
    while len(example_buffer) > 0:
        write_example_buffer()

    # save patch indices for future reference
    patch_indices_filepath = os.path.join(os.path.dirname(tfrecord_filepath), 'patch_indices.pkl')
    with open(patch_indices_filepath, 'wb') as handle:
        pickle.dump(which_patch_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('[o] Saved a record of the patch indices to `%s`.' % patch_indices_filepath)

def make_decode(mode, tf_dtype, buffer_h, buffer_w, eps, clip_ims, indiv_spp=-1):
    """Mode options: 'diff', 'spec', 'comb'."""
    diff_or_comb = mode in {'diff', 'comb'}
    spec_or_comb = mode in {'spec', 'comb'}

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
        if diff_or_comb:
            _features['diffuse'] = tf.FixedLenFeature([], tf.string)
            _features['diffuseVariance'] = tf.FixedLenFeature([], tf.string)
            _features['gt_diffuse'] = tf.FixedLenFeature([], tf.string)
            _features['gt_albedo'] = tf.FixedLenFeature([], tf.string)
        if spec_or_comb:
            _features['specular'] = tf.FixedLenFeature([], tf.string)
            _features['specularVariance'] = tf.FixedLenFeature([], tf.string)
            _features['gt_specular'] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(serialized_example, _features)
        p = {}  # "p" for "parsed"
        for name in _features.keys():
            p[name] = tf.decode_raw(features[name], tf_dtype)
            if indiv_spp > 0 and not name.startswith('gt'):
                p[name] = tf.reshape(p[name], [-1, buffer_h, buffer_w, N_CHANNELS[name]])[:indiv_spp]
            else:
                p[name] = tf.reshape(p[name], [buffer_h, buffer_w, N_CHANNELS[name]])

        # clipping
        if clip_ims:
            if diff_or_comb:
                p['diffuse']    = tf_clip_and_gamma_correct(p['diffuse'])
                p['gt_diffuse'] = tf_clip_and_gamma_correct(p['gt_diffuse'])
            if spec_or_comb:
                p['specular']    = tf_clip_and_gamma_correct(p['specular'])
                p['gt_specular'] = tf_clip_and_gamma_correct(p['gt_specular'])

        # preprocess
        if mode == 'comb':
            p['gt_comb'] = p['gt_diffuse'] + p['gt_specular']
        if diff_or_comb:
            p['diffuse'] = tf_preprocess_diffuse(p['diffuse'], p['albedo'], eps)
            p['gt_diffuse'] = tf_preprocess_diffuse(p['gt_diffuse'], p['gt_albedo'], eps)
            p['diffuseVariance'] = tf_preprocess_diffuse_variance(p['diffuseVariance'], p['albedo'], eps)
        if spec_or_comb:
            p['specularVariance'] = tf_preprocess_specular_variance(
                1.0 + tf.maximum(p['specular'], 0.0), p['specularVariance'])
            p['specular'] = tf_preprocess_specular(p['specular'])
            p['gt_specular'] = tf_preprocess_specular(p['gt_specular'])
        p['depth'], p['depthVariance'] = tf_preprocess_depth(p['depth'], p['depthVariance'])

        variance_features = tf.concat([
            tf_to_rel_variance(p['normalVariance'], p['normal']),
            tf_to_rel_variance(p['albedoVariance'], p['albedo']),
            tf_to_rel_variance(p['depthVariance'],  p['depth'])], axis=-1)

        if mode in {'diff', 'spec'}:
            return (
                p['diffuse'] if mode == 'diff' else p['specular'],
                p['normal'], p['albedo'], p['depth'],
                p['diffuseVariance'] if mode == 'diff' else p['specularVariance'],
                variance_features,
                p['gt_diffuse'] if mode == 'diff' else p['gt_specular'],
            )  # 7 tensors
        elif mode == 'comb':
            return (
                p['diffuse'], p['specular'],
                p['normal'], p['albedo'], p['depth'],
                p['diffuseVariance'], p['specularVariance'], variance_features,
                p['gt_diffuse'], p['gt_specular'], p['gt_comb'],
            )  # 11 tensors

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
                   color_var_weight=1.0, normal_var_weight=1.0, pdf=None):
    """
    Sample NUM_PATCHES (y, x) indices for (PATCH_H, PATCH_W)-sized patches from the given frame.
    As per Bako et al., we will find candidate patches and prune based on color/normal variance.
    Operates in EXR buffer space (as opposed to tensor space).

    We will sample min(NUM_PATCHES, number of nonzero patches), rather than NUM_PATCHES exactly.
    """
    h, w = buffers['normal'].shape[:2]
    y_range = (patch_h // 2, h - patch_h // 2)  # [)
    x_range = (patch_w // 2, w - patch_w // 2)  # [)

    color = clip_and_gamma_correct(buffers['default'])
    c_var = None
    n_var = None

    # 2D PDF ---------------------------------------------------------------

    if pdf is None:
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
    else:
        print('[o] (using error-based sampling map)')

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

        if c_var is not None:
            _c_var = cv2.resize(c_var, qsize)
            imsave(os.path.join(debug_dir, input_id + '_00_color_var.jpg'), _c_var)
        if n_var is not None:
            _n_var = cv2.resize(n_var, qsize)
            imsave(os.path.join(debug_dir, input_id + '_01_normal_var.jpg'), _n_var)

        _pdf = cv2.resize(pdf, qsize)
        imsave(os.path.join(debug_dir, input_id + '_02_pdf.jpg'),             _pdf)
        imsave(os.path.join(debug_dir, input_id + '_03_patches.jpg'),         patches)
        imsave(os.path.join(debug_dir, input_id + '_04_patches_overlay.jpg'), patches_overlay)

    return patch_indices

def write_error_map(error_map, key, filepath):
    error_maps = {}
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as handle:
            error_maps = pickle.load(handle)
    # update with new error map
    error_maps[key] = error_map
    with open(filepath, 'wb') as handle:
        pickle.dump(error_maps, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_sampling_map(error_map, patch_size):
    """Generate a sampling map from a (nonnegative) error map."""
    box_filter  = np.ones((patch_size, patch_size)) / float(patch_size * patch_size)
    # per-window error instead of per-pixel error
    patch_error_r = convolve(error_map[:, :, 0], box_filter, 'same', 'auto')
    patch_error_g = convolve(error_map[:, :, 1], box_filter, 'same', 'auto')
    patch_error_b = convolve(error_map[:, :, 2], box_filter, 'same', 'auto')
    patch_error   = patch_error_r + patch_error_g + patch_error_b
    # invalidate out-of-bounds regions
    y_range = (patch_size // 2, patch_error.shape[0] - patch_size // 2)  # [)
    x_range = (patch_size // 2, patch_error.shape[1] - patch_size // 2)  # [)
    _template = np.zeros_like(patch_error)
    _template[y_range[0]:y_range[1], x_range[0]:x_range[1]] \
        = patch_error[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    pdf = np.maximum(_template, 0.0)
    # normalize to create 2D PDF
    return pdf / np.sum(pdf)

# ===============================================
# PRE/POST-PROCESSING (NUMPY)
# ===============================================

def to_rel_variance(var, _buffer, clip=False):
    """Convert variance to relative variance."""
    rel_var = var / (np.mean(_buffer, axis=-1, keepdims=True) ** 2 + 1e-5)
    if clip:
        rel_var = np.clip(rel_var, 0.0, 1.0)
    return rel_var

def log_transform(im):
    im = np.maximum(im, 0.0)
    return np.log(im + 1.0)

def inv_log_transform(im):
    return np.exp(im) - 1.0

def clip_and_gamma_correct(im):
    assert im.dtype != np.uint8
    return np.clip(im, 0.0, 1.0) ** (1.0 / 2.2)

def preprocess_diffuse(buffers, eps):
    """Factor out albedo. Destructive."""
    buffers['diffuse'] /= buffers['albedo'] + eps
    buffers['diffuse'] = log_transform(buffers['diffuse'])
    mean_albedo = np.mean(buffers['albedo'], axis=-1, keepdims=True)
    buffers['diffuseVariance'] /= np.square(mean_albedo + eps)

def preprocess_specular(buffers):
    """Apply logarithmic transform. Destructive."""
    mean_specular = np.mean(
        1.0 + np.maximum(buffers['specular'], 0.0), axis=-1, keepdims=True)
    buffers['specularVariance'] /= np.square(mean_specular) + 1e-5
    buffers['specular'] = log_transform(buffers['specular'])

def preprocess_depth(buffers):
    """Scale depth to range [0, 1]. Destructive."""
    buffers['depth'] = np.maximum(buffers['depth'], 0.0)
    _max = np.amax(buffers['depth'])
    if _max > 0.0:
        buffers['depth'] /= _max
        buffers['depthVariance'] /= _max ** 2

def compute_buffer_gradients(buffers, indiv_spp=-1):
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

    if indiv_spp > 0:
        def stacked_gradients(_buffer):
            grad_y_tensors, grad_x_tensors = zip(
                *[_image_gradients(_buffer[i]) for i in range(indiv_spp)])
            return np.stack(grad_y_tensors, axis=0), np.stack(grad_x_tensors, axis=0)
        gradient_function = stacked_gradients
    else:
        gradient_function = _image_gradients
    diffuse_grad_y,  diffuse_grad_x  = gradient_function(buffers['diffuse'])
    specular_grad_y, specular_grad_x = gradient_function(buffers['specular'])
    normal_grad_y,   normal_grad_x   = gradient_function(buffers['normal'])
    albedo_grad_y,   albedo_grad_x   = gradient_function(buffers['albedo'])
    depth_grad_y,    depth_grad_x    = gradient_function(buffers['depth'])

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

def make_network_inputs(buffers, clip_ims, eps, indiv_spp=-1):
    """Takes buffers (for a single image, i.e. shapes are HWC)
    and constructs both diffuse and specular KPCN inputs.

    This function does not split the image into patches.
    If such behavior is desired, it can be applied after this function.
    """
    if indiv_spp > 0:
        feature_names = [
            'diffuse',
            'diffuseVariance',
            'specular',
            'specularVariance',
            'albedo',
            'albedoVariance',
            'normal',
            'normalVariance',
            'depth',
            'depthVariance',
        ]
        feature = {}
        for fname in feature_names:
            feature[fname] = []
            for ibuffer in buffers:
                feature[fname].append(ibuffer[fname])
            feature[fname] = np.array(feature[fname])[:indiv_spp]  # all will be (spp, h, w, 3 or 1)
        buffers = feature

    # clip
    if clip_ims:
        buffers['diffuse']  = clip_and_gamma_correct(buffers['diffuse'])
        buffers['specular'] = clip_and_gamma_correct(buffers['specular'])

    # preprocess
    preprocess_diffuse(buffers, eps)
    preprocess_specular(buffers)
    preprocess_depth(buffers)

    var_features = np.concatenate([
        to_rel_variance(buffers['normalVariance'], buffers['normal']),
        to_rel_variance(buffers['albedoVariance'], buffers['albedo']),
        to_rel_variance(buffers['depthVariance'],  buffers['depth']),
    ], axis=-1)

    grad_y, grad_x = compute_buffer_gradients(buffers, indiv_spp=indiv_spp)
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
    return inv_log_transform(out_diffuse) * (albedo + eps)

def postprocess_specular(out_specular):
    """Apply exponential transform."""
    return inv_log_transform(out_specular)

# ===============================================
# PRE/POST-PROCESSING (TENSORFLOW)
# ===============================================

def tf_to_rel_variance(var, _buffer, clip=False):
    """Convert variance to relative variance."""
    rel_var = var / (tf.square(
        tf.reduce_mean(_buffer, axis=-1, keepdims=True)) + 1e-5)
    if clip:
        rel_var = tf.clip_by_value(rel_var, 0.0, 1.0)
    return rel_var

def tf_log_transform(im):
    im = tf.maximum(im, 0.0)
    return tf.log(im + 1.0)

def tf_inv_log_transform(im):
    return tf.exp(im) - 1.0

def tf_clip_and_gamma_correct(im):
    return tf.pow(tf.clip_by_value(im, 0.0, 1.0), 1.0 / 2.2)

def tf_preprocess_diffuse(diffuse, albedo, eps):
    return tf_log_transform(tf.divide(diffuse, albedo + eps))

def tf_preprocess_diffuse_variance(diffuse_variance, albedo, eps):
    mean_albedo = tf.reduce_mean(albedo, axis=-1, keepdims=True)
    return tf.divide(diffuse_variance, tf.square(mean_albedo) + eps)

def tf_preprocess_specular(specular):
    return tf_log_transform(specular)

def tf_preprocess_specular_variance(specular, specular_variance):
    mean_specular = tf.reduce_mean(specular, axis=-1, keepdims=True)
    return tf.divide(specular_variance, tf.square(mean_specular) + 1e-5)

def tf_preprocess_depth(depth, depth_variance):
    depth = tf.maximum(depth, 0.0)
    _max = tf.reduce_max(depth)
    depth = tf.cond(tf.greater(_max, 0.0),
        lambda: depth / _max, lambda: depth)
    depth_variance = tf.cond(tf.greater(_max, 0.0),
        lambda: depth_variance / tf.square(_max), lambda: depth_variance)
    return depth, depth_variance

def tf_postprocess_diffuse(out_diffuse, albedo, eps):
    return tf.multiply(tf_inv_log_transform(out_diffuse), albedo + eps)

def tf_postprocess_specular(out_specular):
    return tf_inv_log_transform(out_specular)

def tf_center_crop(im, y_extent, x_extent):
    _, im_h, im_w, _ = im.get_shape().as_list()
    if y_extent <= im_h:
        y0 = im_h // 2 - y_extent // 2      # [
        y1 = im_h // 2 + y_extent // 2 + 1  # )
    else:
        print('[-] warning: y_extent > im_h')
        y0, y1 = 0, im_h
    if x_extent <= im_w:
        x0 = im_w // 2 - x_extent // 2      # [
        x1 = im_w // 2 + x_extent // 2 + 1  # )
    else:
        print('[-] warning: x_extent > im_w')
        x0, x1 = 0, im_w
    return im[:, y0:y1, x0:x1, :]

def tf_nan_to_num(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def tf_inf_to_num(x):
    return tf.where(tf.is_inf(x), tf.zeros_like(x), x)

# ===============================================
# VISUALIZATION
# ===============================================

def show_multiple(*ims, **kwargs):
    """Plot multiple images in one grid.
    By default, the maximum row length is three.

    Assumes that each image in IMS is either
    an [h, w, c]-array or an [n, h, w, c]-array.
    """
    row_max      = kwargs.get('row_max', max(3, int(ceil(sqrt(len(ims))))))
    batch_max    = kwargs.get('batch_max', 5)  # viz at most 5 examples per batch
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
        nrows *= min(min(len(ims[0].shape), batch_max), ims[0].shape[0])

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
            for k in range(min(min(_nd, batch_max), _im.shape[0])):
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
# ERROR
# ===============================================

def mse(out, gt_out):
    """Mean squared error."""
    return np.mean((out - gt_out) ** 2)

def mrse(out, gt_out):
    """Mean relative squared error."""
    return np.mean(((out - gt_out) ** 2) / (gt_out ** 2 + 1e-2))

def dssim(out, gt_out):
    """Structural dissimilarity."""
    data_range = out.max() - out.min()
    multichannel = len(out.shape) > 2 and out.shape[-1] > 1
    mssim = compare_ssim(out, gt_out, data_range=data_range, multichannel=multichannel)
    return 1.0 - mssim  # also divide by 2?

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

DAY_FORMAT_STR  = '{d} day(s), {h} hour(s), {m} minute(s), {s} second(s)'
HOUR_FORMAT_STR = '{h} hour(s), {m} minute(s), {s} second(s)'
MIN_FORMAT_STR  = '{m} minute(s), {s} second(s)'
SEC_FORMAT_STR  = '{s} second(s)'

def format_seconds(s):
    """Converts S, a float representing some number of seconds,
    into a string detailing DAYS, HOURS, MINUTES, and SECONDS.
    """
    s = int(s)
    days,    s = s // 86400, s % 86400
    hours,   s = s // 3600,  s % 3600
    minutes, s = s // 60,    s % 60

    if days > 0:
        format_str = DAY_FORMAT_STR
    elif hours > 0:
        format_str = HOUR_FORMAT_STR
    elif minutes > 0:
        format_str = MIN_FORMAT_STR
    else:
        format_str = SEC_FORMAT_STR

    return format_str.format(d=days, h=hours, m=minutes, s=s)

def crop_buffers(buffers, y0, y1, x0, x1):
    for k in buffers:
        buffers[k] = buffers[k][y0:y1, x0:x1]

def atoi(text):
    """Source: https://stackoverflow.com/a/5967539."""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Source: https://stackoverflow.com/a/5967539."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]
