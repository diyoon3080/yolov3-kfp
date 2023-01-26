# Kubeflow Pipeline Using YoloV3 TF2 Model

This repository contains python scripts for creating Kubeflow pipeline configuration file using YoloV3 TF2 model. 
Example pipeline files are inside `pipeline-files-yaml` directory for [brain tumor object detection task](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets).

## Usage
### Install `kfp`
```
pip install kfp
```
### Create config.json
```
{
    "train_dataset_url": URL_TO_GOOGLE_DRIVE_TFRECORD_FILE,
    "val_dataset_url": URL_TO_GOOGLE_DRIVE_TFRECORD_FILE,
    "checkpoint_url": URL_TO_GOOGLE_DRIVE_CHECKPOINT_FOLDER,
    "checkpoint_name": NAME_OF_THE_CHECKPOINT_FILE,
    "test_img_url": URL_TO_GOOGLE_DRIVE_TEST_IMG_FILE,
    "model_size": MODEL_SIZE (in pixels),
    "num_classes": NUM_CLASSES,
    "num_epochs": NUM_EPOCHS,
    "class_names": LIST_OF_CLASS_LABELS
}
```
### Create components and connect them into a pipeline
```
python3 run.py
```
### Upload created pipeline configuration file to Kubeflow
Find the `pipeline.yaml` file inside `pipeline-files-yaml` directory and upload it to Kubeflow.

## References
- [yolov3_minimal](https://pypi.org/project/yolov3-minimal/)