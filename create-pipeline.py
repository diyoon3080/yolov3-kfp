import kfp
from kfp import components, dsl

import json

load_train_data_op = components.load_component_from_file('./component-files-yaml/load_train_data_component.yaml')

load_weights_op = components.load_component_from_file('./component-files-yaml/load_weights_component.yaml')

train_model_op = components.load_component_from_file('./component-files-yaml/train_model_component.yaml')

load_test_img_op = components.load_component_from_file('./component-files-yaml/load_test_img_component.yaml')

test_op = components.load_component_from_file('./component-files-yaml/test_component.yaml')

@dsl.pipeline(name='YOLOv3 pipeline')
def yolov3_pipeline():

    with open("config.json") as f:
        config_object = json.load(f)

    train_dataset_url = config_object['train_dataset_url']
    val_dataset_url = config_object['val_dataset_url']
    checkpoint_url = config_object['checkpoint_url']
    checkpoint_name = config_object['checkpoint_name']
    test_img_url = config_object['test_img_url']
    model_size = config_object['model_size']
    num_classes = config_object['num_classes']
    num_epochs = config_object['num_epohcs']
    class_names = config_object['class_names']

    load_data_task = load_train_data_op(train_dataset_url, val_dataset_url)
    load_weights_task = load_weights_op(checkpoint_url)

    train_model_task = train_model_op(
        model_size,
        num_classes,
        num_epochs,
        class_names,
        checkpoint_name,
        load_weights_task.outputs['pretrained_weights'],
        load_data_task.outputs['train_dataset'],
        load_data_task.outputs['val_dataset'],
    )

    load_test_img_task = load_test_img_op(test_img_url)

    test_task = test_op(
        model_size,
        num_classes,
        class_names,
        train_model_task.outputs['trained_weights'],
        load_test_img_task.outputs['input_img']
    )

kfp.compiler.Compiler().compile(yolov3_pipeline, './pipeline-files-yaml/pipeline.yaml')