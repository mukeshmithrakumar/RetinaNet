"""
MIT License

Copyright (c) 2018 Mukesh Mithrakumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import sys


if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.trainer"

from ..models import model_backbone


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument(
        'main_dir',
        help='Path to dataset directory.'
    )
    parser.add_argument(
        'model_in',
        help='The model to convert.'
    )
    parser.add_argument(
        '--backbone',
        help='The backbone of the model to convert.',
        default='resnet50'
    )
    parser.add_argument(
        '--no-nms',
        help='Disables non maximum suppression.',
        dest='nms',
        action='store_false'
    )
    parser.add_argument(
        '--no-class-specific-filter',
        help='Disables class specific filtering.',
        dest='class_specific_filter',
        action='store_false'
    )

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # load and convert model
    model = model_backbone.load_model(args.model_in,
                                      convert=True,
                                      backbone_name=args.backbone,
                                      nms=args.nms,
                                      class_specific_filter=args.class_specific_filter)

    # save model
    model_out_path = os.path.join(args.main_dir, 'keras_retinanet', 'trainer', 'snapshots')
    model_out = os.path.join(model_out_path, '{}_inference.h5'.format(args.model_in))
    model.save(model_out)


if __name__ == '__main__':
    main()
