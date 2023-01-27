import kfp
from kfp import components, dsl

import json

load_train_data_op = components.load_component_from_file('./component-files-yaml/load_train_data_component.yaml')

load_weights_op = components.load_component_from_file('./component-files-yaml/load_weights_component.yaml')

train_model_op = components.load_component_from_file('./component-files-yaml/train_model_component.yaml')

load_test_img_op = components.load_component_from_file('./component-files-yaml/load_test_img_component.yaml')

test_op = components.load_component_from_file('./component-files-yaml/test_component.yaml')

serve_op = components.load_component_from_file('./component-files-yaml/serve_component.yaml')

@dsl.pipeline(name='YOLOv3 pipeline')
def yolov3_pipeline(
    train_dataset_url="https://drive.google.com/file/d/1Sq0bph5QJE5U_x-qu8hUcjgiTONeBDy1/view?usp=sharing",
    val_dataset_url="https://drive.google.com/file/d/172vMkaGKkol2x1juNzjWdEwNZrTyZnvz/view?usp=share_link",
    checkpoint_url="https://drive.google.com/drive/folders/1-C3N6h-CtdojHjEyFvXGXBorDFBo_k4z?usp=share_link",
    checkpoint_name="axial_ckpt.tf",
    test_img_url="https://drive.google.com/file/d/13PzBr8jBjHdt4VMaEcEOkMoCF6VKpmx1/view?usp=share_link",
    model_size='256',
    num_classes='2',
    num_epochs='1',
    class_names='["negative", "positive"]'
):

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

    serve_op(test_task.output)

kfp.compiler.Compiler().compile(yolov3_pipeline, './pipeline-files-yaml/pipeline.yaml')