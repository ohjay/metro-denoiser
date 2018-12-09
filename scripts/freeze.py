# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.tools import optimize_for_inference_lib

# ----------

input_checkpoint = '/home/owen/workspace/metro/scripts/checkpoints/spec/var-13999'

# ----------

prefix = 'diffuse' if 'diff' in input_checkpoint else 'specular'
input_node_names  = ['/'.join([prefix, n]) for n in [
    'inputs/color',
    'inputs/grad_x',
    'inputs/grad_y',
    'inputs/var_color',
    'inputs/var_features',
    'is_training',
    'dropout_keep_prob',
]]
output_node_names = ['%s/out' % prefix]
optimize          = False

# ----------

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, input_checkpoint)

    # export variables to constants
    output_gdef = tf.graph_util.convert_variables_to_constants(
        sess, tf.get_default_graph().as_graph_def(), output_node_names)

    # optimize
    if optimize:
        output_gdef = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def=output_gdef,
            input_node_names=input_node_names,
            output_node_names=output_node_names,
            placeholder_type_enum=dtypes.float32.as_datatype_enum)
        output_gdef = strip_unused_lib.strip_unused(
            input_graph_def=output_gdef,
            input_node_names=input_node_names,
            output_node_names=output_node_names,
            placeholder_type_enum=dtypes.float32.as_datatype_enum)

    # serialize
    output_base = 'frozen-%s.pb' % os.path.basename(input_checkpoint)
    output_path = '/'.join(input_checkpoint.split('/')[:-1] + [output_base])
    with tf.gfile.GFile(output_path, "wb") as f:
        f.write(output_gdef.SerializeToString())
    print('[o] Wrote graph to %s.' % output_path)
    print('[o] %d ops in the final graph.' % len(output_gdef.node))
