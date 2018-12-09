import keras
import argparse
import tensorflow as tf
import numpy as np
import sys
import os
import csv
import pandas as pd

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.trainer"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..models import model_backbone
from ..preprocessing.image import read_image_bgr, preprocess_image, resize_image
from ..models import classifier


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument(
        'main_dir',
        help='Path to dataset directory.'
    )
    parser.add_argument(
        'model_in',
        help="The converted model to evaluate."
             "If training model hasn't been converted for inference, run convert_model first."
    )
    parser.add_argument(
        '--train_type',
        help="Type of predictions you want to make"
             "If you want to train for Visual Reltionship, then type -'vr'."
             "If you want to train for Object Detection, then type -'od'."
             "If you want to train for both, then type -'both'.",
        default='both'
    )
    parser.add_argument(
        '--backbone',
        help='The backbone of the model to convert.',
        default='resnet50'
    )

    return parser.parse_args(args)


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_midlabels(main_dir):
    meta_dir = os.path.join(main_dir, 'challenge2018')
    csv_file = 'challenge-2018-class-descriptions-500.csv'
    boxable_classes_descriptions = os.path.join(meta_dir, csv_file)

    id_to_midlabels = {}
    i = 0
    with open(boxable_classes_descriptions, 'r') as descriptions_file:
        for row in csv.reader(descriptions_file):
            if len(row):
                label = row[0]
                id_to_midlabels[i] = label
                i += 1

    return id_to_midlabels


def get_annotations(base_dir, model):
    id_annotations = dict()
    count = 0
    for img in os.listdir(base_dir):
        try:
            img_path = os.path.join(base_dir, img)
            raw_image = read_image_bgr(img_path)
            image = preprocess_image(raw_image.copy())
            image, scale = resize_image(image, min_side=600, max_side=600)
            height, width, _ = image.shape

            img_id = img.strip('.jpg')

            # run network
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            # boxes in (x1, y1, x2, y2) format
            new_boxes2 = []
            for box in boxes[0]:
                x1_int = round((box[0] / width), 3)
                y1_int = round((box[1] / height), 3)
                x2_int = round((box[2] / width), 3)
                y2_int = round((box[3] / height), 3)
                new_boxes2.extend([x1_int, y1_int, x2_int, y2_int])

            new_list = [new_boxes2[i:i + 4] for i in range(0, len(new_boxes2), 4)]

            annotation = {'cls_label': labels, 'box_values': new_list, 'scores': scores}

            if img_id in id_annotations:
                annotations = id_annotations[img_id]
                annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'boxes': [annotation]}

            count += 1
            print("{0}/99999".format(count))

        except:
            print("Did not evaluate {}".format(img))
            continue

    return id_annotations


def od(id_annotations, main_dir):

    id_to_midlabels = get_midlabels(main_dir)

    try:
        predict = pd.DataFrame.from_dict(id_annotations)
    except:
        print("from dict did not work")

    try:
        predict = pd.DataFrame.from_records(id_annotations)
    except:
        print("from records did not work")

    sub = []
    for k in predict:
        # convert class labels to MID  format by iterating through class labels
        new_clslst = list(map(id_to_midlabels.get, predict[k]['boxes'][0]['cls_label'][0]))

        # copy the scores to the mid labels and create bounding box values
        new_boxlist = []
        for i, mids in enumerate(new_clslst):
            if mids is None:
                break
            else:
                scores = predict[k]['boxes'][0]['scores'][0][i]
                _scorelst = str(mids) + ' ' + str(scores)
                boxval = str(predict[k]['boxes'][0]['box_values'][i]).strip("[]")
                _boxlist = _scorelst + ' ' + boxval
                new_boxlist.append(_boxlist)
            i += 1

        new_boxlist = ''.join(str(new_boxlist)).replace(",", '').replace("'", '').replace("[", '').replace("]", '')

        sub.append(new_boxlist)

    mk_path = os.path.join(main_dir, 'ODSubmissions')
    makedirs(mk_path)
    path = os.path.join(main_dir, 'ODSubmissions')

    print("OD predictions complete")
    with open(path + "od.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for line in sub:
            writer.writerow([line])

    header = ["PredictionString"]
    od_file = pd.read_csv(path + "od.csv", names=header)

    ImageId = []
    for k in predict:
        ImageId.append(k)

    se = pd.Series(ImageId)
    od_file['ImageId'] = se.values

    od_file = od_file[["ImageId", "PredictionString"]]
    od_file.to_csv(path + "submission-od.csv", index=False)

    print("Writing OD Submission file")

    if os.path.isfile(path + 'od.csv'):
        os.unlink(path + 'od.csv')


def relationship_list(new_boxlist, new_scorelist, midlist,LogReg):
    XMin = []
    YMin = []
    XMax = []
    YMax = []

    for idx, i in enumerate(new_boxlist):
        XMin.append(new_boxlist[idx][0])
        YMin.append(new_boxlist[idx][1])
        XMax.append(new_boxlist[idx][2])
        YMax.append(new_boxlist[idx][3])

    if len(midlist) % 2 == 0:
        XMin1 = XMin[:int(len(new_boxlist) / 2)]
        YMin1 = YMin[:int(len(new_boxlist) / 2)]
        XMax1 = XMax[:int(len(new_boxlist) / 2)]
        YMax1 = YMax[:int(len(new_boxlist) / 2)]

        new_scorelist1 = new_scorelist[:int(len(new_scorelist) / 2)]
        midlist1 = midlist[:int(len(midlist) / 2)]

        XMin2 = XMin[int(len(new_boxlist) / 2):]
        YMin2 = YMin[int(len(new_boxlist) / 2):]
        XMax2 = XMax[int(len(new_boxlist) / 2):]
        YMax2 = YMax[int(len(new_boxlist) / 2):]

        new_scorelist2 = new_scorelist[int(len(new_scorelist) / 2):]
        midlist2 = midlist[int(len(midlist) / 2):]

    else:
        XMin1 = XMin[:int(len(new_boxlist) / 2)]
        YMin1 = YMin[:int(len(new_boxlist) / 2)]
        XMax1 = XMax[:int(len(new_boxlist) / 2)]
        YMax1 = YMax[:int(len(new_boxlist) / 2)]

        new_scorelist1 = new_scorelist[:int(len(new_scorelist) / 2)]
        midlist1 = midlist[:int(len(midlist) / 2)]

        XMin2 = XMin[int(len(new_boxlist) / 2) + 1:]
        YMin2 = YMin[int(len(new_boxlist) / 2) + 1:]
        XMax2 = XMax[int(len(new_boxlist) / 2) + 1:]
        YMax2 = YMax[int(len(new_boxlist) / 2) + 1:]

        new_scorelist2 = new_scorelist[int(len(new_scorelist) / 2) + 1:]
        midlist2 = midlist[int(len(midlist) / 2) + 1:]

    vr = pd.DataFrame()

    XMin1_se = pd.Series(XMin1)
    YMin1_se = pd.Series(YMin1)
    XMax1_se = pd.Series(XMax1)
    YMax1_se = pd.Series(YMax1)

    new_scorelist1_se = pd.Series(new_scorelist1)
    midlist1_se = pd.Series(midlist1)

    vr['LabelName1'] = midlist1_se.values
    vr['scores1'] = new_scorelist1_se.values
    vr['XMin1'] = XMin1_se.values
    vr['YMin1'] = YMin1_se.values
    vr['XMax1'] = XMax1_se.values
    vr['YMax1'] = YMax1_se.values

    vr['box_1_length'] = vr['XMax1'] - vr['XMin1']
    vr['box_1_height'] = vr['YMax1'] - vr['YMin1']
    vr['box_1_area'] = vr['box_1_length'] * vr['box_1_height']

    XMin2_se = pd.Series(XMin2)
    YMin2_se = pd.Series(YMin2)
    XMax2_se = pd.Series(XMax2)
    YMax2_se = pd.Series(YMax2)

    new_scorelist2_se = pd.Series(new_scorelist2)
    midlist2_se = pd.Series(midlist2)

    vr['LabelName2'] = midlist2_se.values
    vr['scores2'] = new_scorelist2_se.values
    vr['XMin2'] = XMin2_se.values
    vr['YMin2'] = YMin2_se.values
    vr['XMax2'] = XMax2_se.values
    vr['YMax2'] = YMax2_se.values

    vr['box_2_length'] = vr['XMax2'] - vr['XMin2']
    vr['box_2_height'] = vr['YMax2'] - vr['YMin2']
    vr['box_2_area'] = vr['box_2_length'] * vr['box_2_height']

    vr['confidence'] = (vr['scores1'] + vr['scores2']) / 2.0

    vr["xA"] = vr[["XMin1", "XMin2"]].max(axis=1)
    vr["yA"] = vr[["YMin1", "YMin2"]].max(axis=1)
    vr["xB"] = vr[["XMax1", "XMax2"]].min(axis=1)
    vr["yB"] = vr[["YMax1", "YMax2"]].min(axis=1)

    vr["intersectionarea"] = (vr["xB"] - vr["xA"]) * (vr["yB"] - vr["yA"])
    vr["unionarea"] = vr["box_1_area"] + vr["box_2_area"] - vr["intersectionarea"]
    vr["iou"] = (vr["intersectionarea"] / vr["unionarea"])

    drop_columns = ["intersectionarea", "unionarea", "xA", "yA", "xB", "yB", "box_1_area",
                    "box_2_area", "scores1", "scores2", "box_1_length", "box_1_height",
                    "box_2_length", "box_2_height"]

    vr = vr.drop(columns=drop_columns)

    # replace columns with inf values with nan so I could drop those values
    vr = vr.replace([np.inf, -np.inf], np.nan)
    vr = vr.dropna()

    # drop the ious if its less than zero, it means its without any relationships cause of no intersection
    vr_iou_negative = vr[vr['iou'] < 0]
    vr = vr.drop(vr_iou_negative.index, axis=0)

    vr = vr[['confidence', 'LabelName1', 'XMin1', 'YMin1', 'XMax1',
             'YMax1', 'LabelName2', 'XMin2', 'YMin2', 'XMax2', 'YMax2', 'iou']]

    vr_test = vr[['XMin1', 'YMin1', 'XMax1', 'YMax1', 'XMin2', 'YMin2', 'XMax2',
                  'YMax2', 'iou']]

    try:
        vr_pred = LogReg.predict(vr_test)

        relations_file = {'0': 'at',
                          "1": 'hits',
                          "2": 'holds',
                          "3": 'inside_of',
                          "4": 'interacts_with',
                          "5": 'is',
                          "6": 'on',
                          "7": 'plays',
                          "8": 'under',
                          "9": 'wears'
                          }

        def get_vr(row):
            for c in vr_pred_df.columns:
                if row[c] == 1:
                    return c

        vr_pred_df1 = pd.DataFrame(vr_pred, columns=relations_file)
        vr_pred_df = vr_pred_df1.rename(columns=relations_file)
        vr_pred_df = vr_pred_df.apply(get_vr, axis=1)
        vr['Relationship'] = vr_pred_df.values
        vr = vr.dropna()
        vr = vr.drop(columns='iou')

        vrlst = vr.values.tolist()
        new_vrlst = ''.join(str(vrlst)).replace(",", '').replace("'", '').replace("[", '').replace("]", '')

    except:
        print("EMPTY EVALUATION")
        new_vrlst = ''

    return new_vrlst


def vr(id_annotations, logreg, main_dir):

    id_to_midlabels = get_midlabels(main_dir)

    try:
        predict = pd.DataFrame.from_dict(id_annotations)
    except:
        print("from dict did not work")

    try:
        predict = pd.DataFrame.from_records(id_annotations)
    except:
        print("from records did not work")

    sub = []
    for k in predict:
        counter = 0

        # convert class labels to MID  format by iterating through class labels
        clslst = list(map(id_to_midlabels.get, predict[k]['boxes'][0]['cls_label'][0]))

        new_boxlist = []
        new_scorelist = []
        midlist = []
        empty_imgs = []
        for i, mids in enumerate(clslst):
            if mids is None:
                break
            else:
                scores = predict[k]['boxes'][0]['scores'][0][i]
                val = predict[k]['boxes'][0]['box_values'][i]
                new_scorelist.append(scores)
                midlist.append(mids)
                new_boxlist.append(val)
            i += 1
        counter += 1

        if len(midlist) == 0:
            empty_imgs.append(str(counter) + ':' + str(k))
            new_vrlst = ''

        else:
            new_vrlst = relationship_list(new_boxlist, new_scorelist, midlist, logreg)

        sub.append(new_vrlst)
        print("{0}/99999".format(len(sub)))

    mk_path = os.path.join(main_dir, 'VRSubmissions')
    makedirs(mk_path)
    path = os.path.join(main_dir, 'VRSubmissions')

    with open(path + "vr.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for line in sub:
            writer.writerow([line])

    header = ["PredictionString"]
    vr_file = pd.read_csv(path + "vr.csv", names=header)

    ImageId = []
    for k in predict:
        ImageId.append(k)

    se = pd.Series(ImageId)
    vr_file['ImageId'] = se.values

    vr_file = vr_file[["ImageId", "PredictionString"]]
    vr_file.to_csv(path + "submission-vr.csv", index=False)

    print("Writing VR Submission file")

    if os.path.isfile(path + 'vr.csv'):
        os.unlink(path + 'vr.csv')


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    keras.backend.tensorflow_backend.set_session(get_session())

    keras_retinanet = os.path.join(args.main_dir, 'keras_retinanet', 'trainer', 'snapshots')
    path_to_model = os.path.join(keras_retinanet, '{}.h5'.format(args.model_in))
    base_dir = os.path.join(args.main_dir, 'images', 'test')

    # load the evaluation model
    print('Loading model {}, this may take a second...'.format(args.model_in))
    model = model_backbone.load_model(path_to_model, backbone_name='resnet50')

    print("Starting Evaluation...")

    if args.train_type == 'both':
        id_annotations = get_annotations(base_dir, model)
        print("Evaluation Completed")

        print("Starting Object Detection Prediction")
        od(id_annotations, args.main_dir)

        print("Starting Visual Relationship Bounding Box Classifier Training")
        logreg = classifier.vr_bb_classifier(args.main_dir)

        print("Starting Visual Relationship Bounding Box Prediction")
        vr(id_annotations, logreg, args.main_dir)
        print("Prediction Completed")

    elif args.train_type == 'od':
        id_annotations = get_annotations(base_dir, model)
        print("Evaluation Completed")

        print("Starting Object Detection Prediction")
        od(id_annotations, args.main_dir)
        print("Prediction Completed")

    elif args.train_type == 'vr':
        id_annotations = get_annotations(base_dir, model)
        print("Evaluation Completed")

        print("Starting Visual Relationship Bounding Box Classifier Training")
        logreg = classifier.vr_bb_classifier(args.main_dir)

        print("Starting Visual Relationship Bounding Box Prediction")
        vr(id_annotations, logreg, args.main_dir)
        print("Prediction Completed")

    else:
        raise ValueError('Invalid train type received: {}'.format(args.train_type))


if __name__ == '__main__':
    main()
