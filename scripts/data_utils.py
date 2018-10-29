import Imath
import OpenEXR
import numpy as np
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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(tfrecord_filepath, input_exr_files, gt_exr_files, patches_per_im, patch_size, fp16):
    """Export PATCHES_PER_IM examples for each EXR file.
    Accepts two lists of EXR filepaths with corresponding orderings.

    Handles patch sampling but not preprocessing.
    """
    with tf.python_io.TFRecordWriter(tfrecord_filepath) as writer:
        for input_filepath, gt_filepath in zip(input_exr_files, gt_exr_files):
            input_buffers = read_exr(input_filepath, fp16=fp16)
            input_buffers = stack_channels(input_buffers)
            patch_indices = sample_patches(buffers, patches_per_im, patch_size, patch_size)

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
                writer.write(example.SerializeToString())
            print('[o] Wrote %d examples for %s.' % (len(patch_indices), input_filepath))
        print('[+] Finished writing examples to TFRecord file `%s`.' % tfrecord_filepath)

# ===============================================
# SAMPLING
# ===============================================

def gaussian_filter_wrapper(signal, sigma=1.0):
    """Perform Gaussian blurring, allowing for fp16 dtype."""
    fp16 = signal.dtype == np.float16
    if fp16:
        signal = signal.astype(np.float32)
    signal = gaussian_filter(signal, sigma=1.0)
    if fp16:
        signal = signal.astype(np.float16)
    return signal

def sample_patches(buffers, num_patches, patch_h, patch_w):
    """
    Sample NUM_PATCHES (y, x) indices for (PATCH_H, PATCH_W)-sized patches from the given frame.
    As per Bako et al., we will find candidate patches and prune based on color/normal variance.
    Operates in EXR buffer space (as opposed to tensor space).
    """
    # Define multivariate "PDF"
    pdf = np.squeeze(
        buffers['colorVariance'] + buffers['normalVariance'])  # (h, w)
    pdf = gaussian_filter_wrapper(pdf)  # blur as per Gharbi et al.
    pdf -= np.min(pdf)
    pdf /= np.sum(pdf)  # now [0, 1] and sums to 1
    pdf += 0.1  # so we can multiplicatively tamper with "PDF" later

    h, w = buffers['colorVariance'].shape[:2]
    y_range = (patch_h // 2, h - patch_h // 2)  # [)
    x_range = (patch_w // 2, w - patch_w // 2)  # [)

    patch_indices = []
    while len(patch_indices) < num_patches:
        # Uniformly sample candidate indices
        y = np.random.randint(*y_range)
        x = np.random.randint(*x_range)

        # Reject according to "PDF," adjust "PDF" accordingly
        if pdf[y, x] > np.random.random():
            patch_indices.append([y, x])
            pdf[y, x] *= 0.2
        else:
            pdf[y, x] *= 2.0

    print('[o] Sampled %d patch indices.' % num_patches)
    return np.array(patch_indices)  # (num_patches, 2)

# ===============================================
# PRE/POST-PROCESSING (NUMPY)
# ===============================================

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
    """
    def _three_channel_grad(_buffer):
        grad_ry, grad_rx = np.gradient(_buffer[:, :, 0])
        grad_gy, grad_gx = np.gradient(_buffer[:, :, 1])
        grad_by, grad_bx = np.gradient(_buffer[:, :, 2])

        grad_y = np.stack([grad_ry, grad_gy, grad_by], axis=-1)
        grad_x = np.stack([grad_rx, grad_gx, grad_bx], axis=-1)
        return grad_y, grad_x

    def _one_channel_grad(_buffer):
        grad_y, grad_x = np.gradient(_buffer[:, :, 0])
        grad_y, grad_x = np.expand_dims(grad_y, -1), np.expand_dims(grad_x, -1)
        return grad_y, grad_x

    diffuse_grad_y,  diffuse_grad_x  = _three_channel_grad(buffers['diffuse'])
    specular_grad_y, specular_grad_x = _three_channel_grad(buffers['specular'])
    normal_grad_y,   normal_grad_x   = _three_channel_grad(buffers['normal'])
    albedo_grad_y,   albedo_grad_x   = _three_channel_grad(buffers['albedo'])
    depth_grad_y,    depth_grad_x    = _one_channel_grad(buffers['depth'])

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

def postprocess_diffuse(out_diffuse, albedo, eps):
    """Multiply back albedo."""
    return out_diffuse * (albedo + eps)

def postprocess_specular(out_specular):
    """Apply exponential transform."""
    return np.exp(out_specular) - 1.0

# ===============================================
# VISUALIZATION
# ===============================================

def show_multiple(*ims, **kwargs):
    """Plot multiple images in one grid.
    By default, the maximum row length is three.
    """
    # Determine number of rows and columns
    row_max = kwargs.get('row_max', 3)
    nrows, ncols = len(ims) // row_max, len(ims) % row_max
    if ncols == 0:
        ncols = row_max
    else:
        nrows += 1

    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    for j, im in enumerate(ims):
        _im = im
        while len(_im.shape) > 3:
            _im = _im[0]  # eliminate batch dims
        if _im.dtype == np.float16:
            _im = _im.astype(np.float32)
        ax[j // row_max, j % row_max].imshow(_im)

    plt.show()
