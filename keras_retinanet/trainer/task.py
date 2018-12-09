import argparse
import sys
import os


if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.trainer"

    from ..trainer import model  # Your model.py file.

    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument(
        'main_dir',
        help='Path to dataset directory.'
    )
    oid_parser.add_argument(
        '--version',
        help='The current dataset version is v4.',
        default='challenge2018'
    )
    oid_parser.add_argument(
        '--labels-filter',
        help='A list of labels to filter.',
        type=csv_list,
        default=None
    )
    oid_parser.add_argument(
        '--annotation-cache-dir',
        help='Path to store annotation cache.',
        default='.'
    )
    oid_parser.add_argument(
        '--parent-label',
        help='Use the hierarchy children of this label.',
        default=None
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--snapshot',
        help='Resume training from a snapshot.'
    )
    group.add_argument(
        '--imagenet-weights',
        help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
        action='store_const',
        const=True,
        default=True
    )
    group.add_argument(
        '--weights',
        help='Initialize the model with weights from a file.'
    )
    group.add_argument(
        '--no-weights',
        help='Don\'t initialize the model with any weights.',
        dest='imagenet_weights',
        action='store_const',
        const=False
    )

    parser.add_argument(
        '--backbone',
        help='Backbone model used by retinanet.',
        default='resnet50',
        type=str
    )
    parser.add_argument(
        '--batch-size',
        help='Size of the batches.',
        default=1,
        type=int
    )
    parser.add_argument(
        '--gpu',
        help='Id of the GPU to use (as reported by nvidia-smi).'
    )
    parser.add_argument(
        '--multi-gpu',
        help='Number of GPUs to use for parallel processing.',
        type=int,
        default=1)
    parser.add_argument(
        '--multi-gpu-force',
        help='Extra flag needed to enable (experimental) multi-gpu support.',
        action='store_true'
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.',
        type=int,
        default=50
    )
    parser.add_argument(
        '--steps',
        help='Number of steps per epoch.',
        type=int,
        default=100000
    )
    parser.add_argument(
        '--snapshot-path',
        help="Path to store snapshots of models during training (defaults to \'./snapshots\')",
        default='./snapshots'
    )
    parser.add_argument(
        '--tensorboard-dir',
        help='Log directory for Tensorboard output',
        default='./logs'
    )
    parser.add_argument(
        '--no-snapshots',
        help='Disable saving snapshots.',
        dest='snapshots',
        action='store_false'
    )
    parser.add_argument(
        '--no-evaluation',
        help='Disable per epoch evaluation.',
        dest='evaluation',
        action='store_false'
    )
    parser.add_argument(
        '--freeze-backbone',
        help='Freeze training of backbone layers.',
        action='store_true'
    )
    parser.add_argument(
        '--random-transform',
        help='Randomly transform image and annotations.',
        action='store_true'
    )
    parser.add_argument(
        '--image-min-side',
        help='Rescale the image so the smallest side is min_side.',
        type=int,
        default=600
    )
    parser.add_argument(
        '--image-max-side',
        help='Rescale the image if the largest side is larger than max_side.',
        type=int,
        default=600
    )

    args = parser.parse_args()

    # Run the training job
    model.train(args)
