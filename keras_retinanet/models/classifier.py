import pandas as pd
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def vr_bb_classifier(main_dir):

    path = os.path.join(main_dir, 'challenge2018')
    train_file = "relationship_triplets_annotations.csv"

    train = pd.read_csv(path + train_file)

    train['box1length'] = train['XMax1'] - train['XMin1']
    train['box2length'] = train['XMax2'] - train['XMin2']
    train['box1height'] = train['YMax1'] - train['YMin1']
    train['box2height'] = train['YMax2'] - train['YMin2']

    train['box1area'] = train['box1length'] * train['box1height']
    train['box2area'] = train['box2length'] * train['box2height']

    train["xA"] = train[["XMin1", "XMin2"]].max(axis=1)
    train["yA"] = train[["YMin1", "YMin2"]].max(axis=1)
    train["xB"] = train[["XMax1", "XMax2"]].min(axis=1)
    train["yB"] = train[["YMax1", "YMax2"]].min(axis=1)

    train["intersectionarea"] = (train["xB"] - train["xA"]) * (train["yB"] - train["yA"])
    train["unionarea"] = train["box1area"] + train["box2area"] - train["intersectionarea"]
    train["iou"] = (train["intersectionarea"] / train["unionarea"])

    drop_columns = ["ImageID", "box1length", "box2length", "box1height",
                    "box2height", "intersectionarea", "unionarea", "xA", "yA",
                    "xB", "yB", "box1area", "box2area"]
    train = train.drop(columns=drop_columns)

    train = train[['LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2',
                   'XMax2', 'YMin2', 'YMax2', 'iou', 'RelationshipLabel']]

    train = pd.get_dummies(train, columns=["RelationshipLabel"])

    COLUMN_NAMES = {"RelationshipLabel_at": "at",
                    "RelationshipLabel_hits": "hits",
                    "RelationshipLabel_holds": "holds",
                    "RelationshipLabel_inside_of": "inside_of",
                    "RelationshipLabel_interacts_with": "interacts_with",
                    "RelationshipLabel_is": "is",
                    "RelationshipLabel_on": "on",
                    "RelationshipLabel_plays": "plays",
                    "RelationshipLabel_under": "under",
                    "RelationshipLabel_wears": "wears",
                    }

    train = train.rename(columns=COLUMN_NAMES)

    X = train[['XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2',
               'XMax2', 'YMin2', 'YMax2', 'iou']]

    y = train[['at', 'hits', 'holds', 'inside_of', 'interacts_with',
               'is', 'on', 'plays', 'under', 'wears']]

    print("Training VR Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=25)

    forest = RandomForestClassifier(n_estimators=500,
                                    verbose=1)
    LogReg = MultiOutputClassifier(forest).fit(X_train, y_train)

    # y_pred = LogReg.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print("VR Classifier Training Complete")

    return LogReg

