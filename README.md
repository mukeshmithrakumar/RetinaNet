<p align="center"><img width="40%" src="logo/share2.jpg" /></p>

<h1 align="center">Keras RetinaNet</h1>

<img alt="Not Maintained" src="https://img.shields.io/badge/Maintained%3F-no-red.svg" />

<p align="center">
    <a href="https://github.com/ellerbrock/open-source-badges/">
        <img src="https://badges.frapsoft.com/os/v2/open-source.png?v=103" alt="Open Source Love">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/github/license/mashape/apistatus.svg" alt="GitHub">
    </a>
    <a href="https://www.python.org/downloads/release/python-360/">
        <img src="https://img.shields.io/badge/Python-3.6-blue.svg" alt="Python 3.6">
    </a>
    <a href="https://github.com/mukeshmithrakumar/RetinaNet/stargazers">
        <img src="https://img.shields.io/github/stars/mukeshmithrakumar/RetinaNet.svg?colorA=orange&colorB=orange&logo=github&label=Stars"
            alt="GitHub Stars">
    </a>
    <a href="https://www.linkedin.com/in/mukesh-mithrakumar/">
        <img src="https://img.shields.io/badge/LinkedIn-blue.svg?logo=#0077B5" alt="LinkedIn">
    </a>
</p>


<h2 align="center">What is it :question:</h2>

This is the Keras implementation of RetinaNet for object detection as described in
[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v:


##### Object Detection:
The RetinaNet used is a single, unified network composed of a resnet50 backbone network and two task-specific
subnetworks. The backbone is responsible for computing a convolution feature map over an entire input image and is
an off-the-self convolution network. The first subnet performs classification on the backbones output; the second
subnet performs convolution bounding box regression.
The RetinaNet is a good model for object detection but getting it to work was a challenge. I underestimated the high
number of classes and the size of the data set but was still able to land a bronze medal (Top 20%) among 450
competitors with some tweaks. The benchmark file is added for reference with the local score for predictions and
the parameter used.

##### Visual Relationship:
I focused on Object detection and used a simple multi class linear regressor for relationship prediction. Unlike the
usual approach of using a LSTM, I experimented with a Random Forest Classifier and a Multi Output Classifier from
sklearn just to prove LSTM doesn't have much intelligence behind it and it was just a statistical tool. And the
local classification scores proved I was right with giving me an accuracy greater than 90%. And since my visual
relationship was based on how good my object detector performed I was not able to get a better score but with this
model I was able to land a bronze model (Top 30%) among 230 competitors.

##### Lessons Learned with Tips:
1. Not to threshold the predictions and leave the low confidence predictions in the submission file.
Because of the way average precision works, you cannot be penalised for adding additional false positives
with a lower confidence than all your other predictions, however you can still improve your recall if you
find additional objects that weren’t previously detected.
2. The number of steps and epochs, due to the number of images in the train set, having a balanced number of steps
and epochs is very important and more important than that is to take all these classes and divide it into bins.
Where each bin is occupied by classes with similar frequency in the data set to prepare proper epoch.
3. When running the training for the classes, to make sure that each class (within an epoch)
has similar number of occurrences by implementing a sampler to do this work.

<h2 align="center">:clipboard: Getting Started</h2>

The build was made for the Google AI Object Detection and Visual Relationship Kaggle challenge so if you are using
this project on Googles' Open Image data set follow the instructions below to run the module. Also the code is written
in such a way that you can take individual modules to build a custom model as per your needs. So when you install the
model, make sure you turn the imports into absolute imports or follow the Folder Structure shown below.

### :dvd: Software Prerequisites

- keras
- keras-resnet
- tensorflow
- pandas
- numpy
- pillow
- opencv
- sklearn

### :computer: Hardware Prerequisites
The code was initially run on a NVIDIA GeForce GTX 1050 Ti but the model exploded since for the Open Image data set
consisted of 1,743,042 Images and 500 classes with 12,195,144 bounding boxes and the image size was resized to
600 by 600. Resizing the images could have solved the issue but did not try it. Instead the code was run on a
NVIDIA Tesla K80 and the model worked fine and to convert the training model to a inference model NVIDIA Tesla P100
was used. So I would recommend a K80 or a higher version of GPU.

### :blue_book: Folder Structure

```
main_dir
- challenge2018 (The folder containing data files for the challenge)
- images
    - train (consists of the train images)
    - test (consists of the test images)
- keras_retinanet (keras retinanet package)
    - callbacks
        - callbacks.py
    - models
        - classifier.py
        - model_backbone.py
        - resnet.py
        - retinanet.py
    - preprocessing
        - generator.py
        - image.py
        - open_images.py
    - trainer
        - convert_model.py
        - evaluate.py
        - model.py
        - task.py
    - utils
        - anchors.py
        - clean.py
        - freeze.py
        - initializers.py
        - layers.py
        - losses.py
```

<h2 align="center">:hourglass: Train</h2>

Run the ```task.py``` from the trainer folder.

#### Usage
```
task.py main_dir(path/to/main directory) dataset_type(oid)
```

<h2 align="center">:watch: Test</h2>

First run the ```convert_model.py``` to convert the training model to inference model.
Then run the ```evaluate.py``` for evaluation. Evaluation is defaulted for both object detection and visual
relationship identification, to select between the object detection and the visual relationship identification
add 'od' or 'vr' when calling the ```evaluate.py```

#### Usage
```
convert_model.py main_dir(path/to/main directory) model_in(model name to be used to convert)
evaluate.py main_dir(path/to/main directory) model_in(model name to be used for evaluation)
```

<h2 align="center">:page_facing_up: Documentation</h2>


callbacks.py:
- CALLED: at model.py by the create callbacks function
- DOES: returns a set of callbacks used for training

classifier.py:
- CALLED: at evaluate.py by the main function
- DOES: returns a Logistic Regression regressor for visual relationship prediction

model_backbone.py:
* CALLED: at model.py by the train function
* DOES: Load the retinanet model using the correct backbone.

resnet.py:
* CALLED: at model_backbone.py by the backbone function
* DOES: Constructs a retinanet model using a resnet backbone.

retinanet.py:
* CALLED: at resnet.py by the resnet_retinanet function
* DOES: Construct a RetinaNet model on top of a backbone

generator.py:
* CALLED: at open_images.py by the OpenImagesGenerator class
* DOES: creates a train and validation generator for open_images.py processing

image.py:
* CALLED: at generator.py by the Generator class
* DOES: transformations and pre processing on the images

open_images.py:
* CALLED: at model.py by the create_generators function
* DOES: returns train and validation generators

convert_model.py:
* CALLED: stand alone file to convert the train model to inference model
* DOES: converts a train model to inference model

evaluate.py:
* CALLED: stand alone evaluation file
* DOES: object and visual relationship detection and identification

model.py:
* CALLED: at task.py
* DOES: the training

task.py:
* CALLED: stand alone file to be called to start training
* DOES: initiates the training

anchors.py:
* CALLED: at generator.py
* DOES: Generate anchors for bbox detection

clean.py:
* CALLED: stand alone file
* DOES: creates ddata files based on the downloaded train and test images

freeze.py:
* CALLED: at model.py by the create_models function
* DOES: freeze layers for training

initializers.py:
* CALLED: at retinanet.py
* DOES: Applies a prior probability to the weights

layers.py:
* CALLED: at retinanet.py
* DOES: Keras layer for filtering detections

losses.py:
* CALLED: at model.py by the create_models function
* DOES: calculate the focal and smooth_l1 losses

<h2 align="center">:alien: Authors</h2>

* **Mukesh Mithrakumar** - *Initial work* - [Keras_RetinaNet](https://github.com/mukeshmithrakumar/)

<h2 align="center">:key: License</h2>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

<h2 align="center">:loudspeaker: Acknowledgments</h2>

* Inspiration from Fizyr Keras RetinaNet
