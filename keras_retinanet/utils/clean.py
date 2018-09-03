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


# make a list with the downloaded images
import pandas as pd
import os

path_to_datafiles = "/home/mukeshmithrakumar/googleai/gcmount/challenge2018/"

# code to load the train annotation box
print("Reading challenge-2018-train-annotations-bbox.csv ....")
challenge_2018_train_annotations_bbox = pd.read_csv(path_to_datafiles + "challenge-2018-train-annotations-bbox2.csv")
challenge_2018_train_annotations_bbox = pd.DataFrame(challenge_2018_train_annotations_bbox)
print("challenge_2018_train_annotations_bbox shape:", challenge_2018_train_annotations_bbox.shape)

# code to load the validation annotation box
print("Reading challenge-2018-image-ids-valset-od.csv ....")
challenge_2018_image_ids_valset_od = pd.read_csv(path_to_datafiles + "challenge-2018-image-ids-valset-od2.csv")
challenge_2018_image_ids_valset_od = pd.DataFrame(challenge_2018_image_ids_valset_od)
print("challenge_2018_image_ids_valset_od shape:", challenge_2018_image_ids_valset_od.shape)

# goes to the directory of the train/val images and creates a list of the downloaded images
directory = "/home/mukeshmithrakumar/googleai/gcmount/images/train/train"
downloaded_list = []
print("Parsing downloaded files ....")
for filename in os.listdir(directory):
    downloaded_list.append(filename)
print("downloaded files: ", len(downloaded_list))

# strips the imgs of the .jpg tag, see if you can add it to the for loop above
downloaded_list = [imgs.strip('.jpg') for imgs in downloaded_list]

# create a new df with descriptions from annotations for the downloaded images
print("Creating new dataframes ....")
train_annotations_bbox_downloaded_df_train = challenge_2018_train_annotations_bbox[
    challenge_2018_train_annotations_bbox['ImageID'].isin(downloaded_list)]

val_annotations_bbox_downloaded_df_train = challenge_2018_image_ids_valset_od[
    challenge_2018_image_ids_valset_od['ImageID'].isin(downloaded_list)]

print("challenge-2018-train-annotations-bbox shape:", train_annotations_bbox_downloaded_df_train.shape)
print("challenge-2018-image-ids-valset-od shape:", val_annotations_bbox_downloaded_df_train.shape)

# exported the data to csv
print("Exporting the csv files ....")
train_annotations_bbox_downloaded_df_train.to_csv(path_to_datafiles
                                                  + 'challenge-2018-train-annotations-bbox.csv', index=False)
val_annotations_bbox_downloaded_df_train.to_csv(path_to_datafiles
                                                + 'challenge-2018-image-ids-valset-od.csv', index=False)
