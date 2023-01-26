from kfp import components
from kfp.components import InputPath, OutputPath

from typing import List

def test(
    model_size: int,
    num_classes: int,
    class_names: List[str],
    trained_weights: InputPath('Weights'),
    input_img: InputPath('jpg'),
    output_img: OutputPath('jpg')
):
    from yolov3_minimal import transform_images, draw_outputs, YoloV3
    import tensorflow as tf
    import cv2
    import os

    SIZE = model_size
    NUM_CLASSES = num_classes

    model = YoloV3(SIZE, classes=NUM_CLASSES)
    model.load_weights(trained_weights+'/trained_weights.tf').expect_partial()
    print('trained weights loaded')

    img_raw = tf.image.decode_image(open(input_img, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, SIZE)

    boxes, scores, classes, nums = model(img)
    print('inference done')

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    os.mkdir(output_img)
    cv2.imwrite(output_img+'/output.jpg', img)

components.create_component_from_func(
    test,
    output_component_file='./component-files-yaml/test_component.yaml',
    packages_to_install=['yolov3-minimal']
)