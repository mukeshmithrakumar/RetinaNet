from __future__ import print_function

from ..utils.anchors import compute_overlap
import keras
import os
import cv2
import numpy as np
import warnings


class RedirectModel(keras.callbacks.Callback):
    """
    Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


"""
Generated using:
```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
shuffle(colors)
pprint(colors)
```
"""
colors = [
    [31, 0, 255],
    [0, 159, 255],
    [255, 95, 0],
    [255, 19, 0],
    [255, 0, 0],
    [255, 38, 0],
    [0, 255, 25],
    [255, 0, 133],
    [255, 172, 0],
    [108, 0, 255],
    [0, 82, 255],
    [0, 255, 6],
    [255, 0, 152],
    [223, 0, 255],
    [12, 0, 255],
    [0, 255, 178],
    [108, 255, 0],
    [184, 0, 255],
    [255, 0, 76],
    [146, 255, 0],
    [51, 0, 255],
    [0, 197, 255],
    [255, 248, 0],
    [255, 0, 19],
    [255, 0, 38],
    [89, 255, 0],
    [127, 255, 0],
    [255, 153, 0],
    [0, 255, 255],
    [0, 255, 216],
    [0, 255, 121],
    [255, 0, 248],
    [70, 0, 255],
    [0, 255, 159],
    [0, 216, 255],
    [0, 6, 255],
    [0, 63, 255],
    [31, 255, 0],
    [255, 57, 0],
    [255, 0, 210],
    [0, 255, 102],
    [242, 255, 0],
    [255, 191, 0],
    [0, 255, 63],
    [255, 0, 95],
    [146, 0, 255],
    [184, 255, 0],
    [255, 114, 0],
    [0, 255, 235],
    [255, 229, 0],
    [0, 178, 255],
    [255, 0, 114],
    [255, 0, 57],
    [0, 140, 255],
    [0, 121, 255],
    [12, 255, 0],
    [255, 210, 0],
    [0, 255, 44],
    [165, 255, 0],
    [0, 25, 255],
    [0, 255, 140],
    [0, 101, 255],
    [0, 255, 82],
    [223, 255, 0],
    [242, 0, 255],
    [89, 0, 255],
    [165, 0, 255],
    [70, 255, 0],
    [255, 0, 172],
    [255, 76, 0],
    [203, 255, 0],
    [204, 0, 255],
    [255, 0, 229],
    [255, 133, 0],
    [127, 0, 255],
    [0, 235, 255],
    [0, 255, 197],
    [255, 0, 191],
    [0, 44, 255],
    [50, 255, 0]
]


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes.
                            By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label).
        color         : The color of the boxes.
                        By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label = a[4]
        c = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    while True:
        for i in range(generator.size()):
            try:
                raw_image = generator.load_image(i)
                image = generator.preprocess_image(raw_image.copy())
            except:
                break
            image, scale = generator.resize_image(image)

            # run network
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes = boxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]
            image_detections = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

            if save_path is not None:
                draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
                draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

                cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

            print('{}/{}'.format(i + 1, generator.size()), end='\r')

        return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self, generator, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None,
                 tensorboard=None, verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.verbose = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            if num_annotations > 0:
                present_classes += 1
                precision += average_precision
        self.mean_ap = precision / present_classes

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
