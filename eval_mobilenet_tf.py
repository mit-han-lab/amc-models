# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import os
import sys
# add slim to PATH
home = os.getenv("HOME")
sys.path.insert(0, os.path.join(home, 'models/research/slim/'))
sys.path.insert(0, '..')

from datasets import dataset_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'profile', '0.5time', 'The profile to use for MobileNetV1 (1.0/0.75/0.5flops/0.5time).')

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

# tf.app.flags.DEFINE_string(
#     'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/ssd/dataset/tf-imagenet', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 1,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')


tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


from models import mobilenet_v1_tf as mobilenet_v1
from models.mobilenet_v1_tf import Conv, DepthSepConv

profiles = {
    # '1.0': [32, 64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024],
    # '0.75': [24, 48, (96, 2), 96, (192, 2), 192, (384, 2), 384, 384, 384, 384, 384, (768, 2), 768],
    '0.5flops': [24, 48, (96, 2), 80, (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752],
    '0.5time': [16, 48, (88, 2), 80, (192, 2), 168, (336, 2), 360, 360, 352, 352, 368, (768, 2), 680],
}

checkpoints = {
    '0.5flops': './checkpoints/tf/0.5flops/',
    '0.5time': './checkpoints/tf/0.5time/',
}


def build_conv_defs():
    profile = profiles[FLAGS.profile]
    conv_defs = []
    conv_defs.append(Conv(kernel=[3, 3], stride=2, depth=profile[0]))
    for p in profile[1:]:
        if type(p) == tuple:
            conv_defs.append(DepthSepConv(kernel=[3, 3], stride=p[1], depth=p[0]))
        else:
            conv_defs.append(DepthSepConv(kernel=[3, 3], stride=1, depth=p))
    dm = 1
    return conv_defs, dm


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)


        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        def preprocess_for_eval(image, height=224, width=224,
                                central_fraction=0.875, scope=None):
            assert height == width  # square input for imagenet

            def proc_func(x):
                x = (x * 255).astype('uint8')

                import PIL
                import numpy as np
                x = PIL.Image.fromarray(x)

                size = int(height / central_fraction)
                interpolation = PIL.Image.BILINEAR
                w, h = x.size
                if (w <= h and w == size) or (h <= w and h == size):
                    pass
                if w < h:
                    ow = size
                    oh = int(size * h / w)
                    x = x.resize((ow, oh), interpolation)
                else:
                    oh = size
                    ow = int(size * w / h)
                    x = x.resize((ow, oh), interpolation)

                w, h = x.size
                th, tw = height, width
                i = int(round((h - th) / 2.))
                j = int(round((w - tw) / 2.))

                x = x.crop((j, i, j+tw, i+th))

                return np.array(x).astype('float32') / 255

            with tf.name_scope(scope, 'eval_image', [image, height, width]):
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

                image = tf.py_func(proc_func, [image], tf.float32)
                image.set_shape([224, 224, 3])

                means = [0.485, 0.456, 0.406]
                stds = [0.229, 0.224, 0.225]
                channels = tf.split(axis=2, num_or_size_splits=3, value=image)
                for i in range(3):
                    channels[i] -= means[i]
                    channels[i] /= stds[i]
                return tf.concat(axis=2, values=channels)

        eval_image_size = FLAGS.eval_image_size

        image = preprocess_for_eval(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        conv_defs, dm = build_conv_defs()
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, _ = mobilenet_v1.mobilenet_v1(images, 1000, dropout_keep_prob=0.8, is_training=False,
                                                  conv_defs=conv_defs, depth_multiplier=dm)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        tf.logging.info('Evaluating %s' % checkpoints[FLAGS.profile])

        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=checkpoints[FLAGS.profile],
            logdir=os.path.join('./eval', checkpoints[FLAGS.profile]),
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            max_number_of_evaluations=1,
        )


if __name__ == '__main__':
    tf.app.run()
