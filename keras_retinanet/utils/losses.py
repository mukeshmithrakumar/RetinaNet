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

import keras
import tensorflow as tf


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the classification focal loss.
    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.
        As defined in https://arxiv.org/abs/1708.02002
        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 regression loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the
            anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state = y_true[:, :, 4]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
