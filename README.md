# Kubeflow Pipeline Using YoloV3 TF2 Model

This repository contains python scripts for creating Kubeflow pipeline configuration file using YoloV3 TF2 model. 

## Usage
### Install `kfp`
```
pip install kfp
```
### Create components and connect them into a pipeline
```
python3 run.py
```
### Upload created pipeline configuration file to Kubeflow
Find the `pipeline.yaml` file inside `pipeline-files-yaml` directory and upload it to Kubeflow.
### Provide necessary pipeline parameters
```
train_dataset_url: Google Drive url to the tfrecord file for train dataset
val_dataset_url: Google Drive url to the tfrecord file for val dataset
checkpoint_url: Google Drive url to the pretrained checkpoint directory
checkpoint_name: name of the checkpoint file inside the checkpoint directory
test_img_url: Google Drive url to the image to be used for the test step
model_size: size of the YoloV3 model
num_classes: number of the classes of the labels
num_epochs: number of epochs for training
class_names: list of class labels
```

## References
- [yolov3_minimal](https://pypi.org/project/yolov3-minimal/)
