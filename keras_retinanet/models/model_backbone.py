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


import keras.models


class Backbone(object):
    """ This class stores additional information on backbones.
    """

    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        from ..utils import layers
        from ..utils import losses
        from ..utils import initializers
        self.custom_objects = {
            'UpsampleLike': layers.UpsampleLike,
            'PriorProbability': initializers.PriorProbability,
            'RegressBoxes': layers.RegressBoxes,
            'FilterDetections': layers.FilterDetections,
            'Anchors': layers.Anchors,
            'ClipBoxes': layers.ClipBoxes,
            '_smooth_l1': losses.smooth_l1(),
            '_focal': losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from ..models.resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50', convert=False, nms=True, class_specific_filter=True):
    """ Loads a retinanet model using the correct custom objects.
    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.
        convert               : Boolean, whether to convert the model to an inference model.
        nms                   : Boolean, whether to add NMS filtering to the converted model.
                                Only valid if convert=True.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
    # Returns
        A keras.models.Model object.
    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """

    model = keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
    if convert:
        from ..models.retinanet import retinanet_bbox
        print("Starting to convert model...")
        model = retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter)

    return model
