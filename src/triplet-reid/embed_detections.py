#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
import common
from duke_utils import *
import scipy.io as sio
import functools

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset_path', default='F:/DukeMTMC/', 
    help='Dataset root.')

parser.add_argument(
    '--detections_path', default=None, required=True, 
    help='Detections .mat file')

parser.add_argument(
    '--filename', default=None, required=True,
    help='Name of the HDF5 file in which to store the embeddings')

# Optional

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel data loading.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')



parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', choices=['center', 'avgpool', 'five'], default=None,
    help='When this flag is provided, crop augmentation is performed.'
         '`avgpool` means the full image at the precrop size is used and '
         'the augmentation is performed by the average pooling. `center` means'
         'only the center crop is used and `five` means the four corner and '
         'center crops are used. When not provided, by default the image is '
         'resized to network input size.')

parser.add_argument(
    '--aggregator', choices=AGGREGATORS.keys(), default=None,
    help='The type of aggregation used to combine the different embeddings '
         'after augmentation.')

parser.add_argument(
    '--quiet', action='store_true', default=False,
    help='Don\'t be so verbose.')




def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    return images, tf.stack([fid]*2), tf.stack([pid]*2)


def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.floor_div(crop_margin, 2)
        bottom_right = tf.add(top_left, crop_size)
    center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left     = image[:-crop_margin[0], :-crop_margin[1]]
    top_right    = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right


def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    # Load the args from the original experiment.
    args_file = os.path.join(args.experiment_root, 'args.json')

    if os.path.isfile(args_file):
        if not args.quiet:
            print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # Add arguments from training.
        for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

        # A couple special-cases and sanity checks
        if (args_resumed['crop_augment']) == (args.crop_augment is None):
            print('WARNING: crop augmentation differs between training and '
                  'evaluation.')
        args.image_root = args.image_root or args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))

    # Check a proper aggregator is provided if augmentation is used.
    if args.flip_augment or args.crop_augment == 'five':
        if args.aggregator is None:
            print('ERROR: Test time augmentation is performed but no aggregator'
                  'was specified.')
            exit(1)
    else:
        if args.aggregator is not None:
            print('ERROR: No test time augmentation that needs aggregating is '
                  'performed but an aggregator was specified.')
            exit(1)

    if not args.quiet:
        print('Evaluating using the following parameters:')
        for key, value in sorted(vars(args).items()):
            print('{}: {}'.format(key, value))

    # Load the data from the CSV file.

    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    # Load detections
    matfile = sio.loadmat(args.detections_path)
    detections = matfile['detections']
    num_detections = detections.shape[0]

    # Setup a tf Dataset generator
    generator = functools.partial(detections_generator, args.dataset_path, detections, net_input_size[0], net_input_size[1])
    dataset = tf.data.Dataset.from_generator(generator, tf.float32, tf.TensorShape([net_input_size[0], net_input_size[1], 3]))
    
    modifiers = ['original']
    if args.flip_augment:
        dataset = dataset.map(flip_augment)
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

    if args.crop_augment == 'center':
        dataset = dataset.map(lambda im, fid, pid:
            (five_crops(im, net_input_size)[0], fid, pid))
        modifiers = [o + '_center' for o in modifiers]
    elif args.crop_augment == 'five':
        dataset = dataset.map(lambda im, fid, pid:
            (tf.stack(five_crops(im, net_input_size)), [fid]*5, [pid]*5))
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for o in modifiers for m in [
            '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
    elif args.crop_augment == 'avgpool':
        modifiers = [o + '_avgpool' for o in modifiers]
    else:
        modifiers = [o + '_resize' for o in modifiers]
    

    # Group it back into PK batches.
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(args.batch_size)
    images = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)

    with h5py.File(args.filename, 'w') as f_out, tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        if args.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        else:
            checkpoint = os.path.join(args.experiment_root, args.checkpoint)
        if not args.quiet:
            print('Restoring from checkpoint: {}'.format(checkpoint))
        tf.train.Saver().restore(sess, checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros(
            (num_detections * len(modifiers), args.embedding_dim), np.float32)
        
        for start_idx in count(step=args.batch_size):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(
                        start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        if not args.quiet:
            print("Done with embedding, aggregating augmentations...", flush=True)

        if len(modifiers) > 1:
            # Pull out the augmentations into a separate first dimension.
            emb_storage = emb_storage.reshape(len(data_fids), len(modifiers), -1)
            emb_storage = emb_storage.transpose((1,0,2))  # (Aug,FID,128D)

            # Store the embedding of all individual variants too.
            emb_dataset = f_out.create_dataset('emb_aug', data=emb_storage)

            # Aggregate according to the specified parameter.
            emb_storage = AGGREGATORS[args.aggregator](emb_storage)

        # Store the final embeddings.
        emb_dataset = f_out.create_dataset('emb', data=emb_storage)

        # Store information about the produced augmentation and in case no crop
        # augmentation was used, if the images are resized or avg pooled.
        f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))


if __name__ == '__main__':
    main()
