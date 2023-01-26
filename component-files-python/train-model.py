from kfp import components
from kfp.components import InputPath, OutputPath
from typing import List

def train_model(
    model_size: int,
    num_classes: int,
    num_epohcs: int,
    class_names: List[str],
    checkpoint_name: str,
    pretrained_weights: InputPath('Weights'),
    train_dataset: InputPath('Dataset'),
    val_dataset: InputPath('Dataset'),
    trained_weights: OutputPath('Weights')
):
    from yolov3_minimal import load_tfrecord_dataset, transform_images, transform_targets, YoloV3, YoloLoss, freeze_all
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau,
        EarlyStopping,
        ModelCheckpoint
    )

    SIZE = model_size
    NUM_CLASSES = num_classes
    NUM_EPOCHS = num_epohcs
    LEARNING_RATE = 1e-3

    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)],np.float32) / 416
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    train_dataset = load_tfrecord_dataset(train_dataset, class_names, SIZE)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(8)
    train_dataset = train_dataset.map(lambda x, y: (
        transform_images(x, SIZE),
        transform_targets(y, anchors, anchor_masks, SIZE)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_tfrecord_dataset(val_dataset, class_names, SIZE)
    val_dataset = val_dataset.batch(8)
    val_dataset = val_dataset.map(lambda x, y: (
        transform_images(x, SIZE),
        transform_targets(y, anchors, anchor_masks, SIZE)))
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    model = YoloV3(SIZE, classes=NUM_CLASSES, training=True)
    model.load_weights(pretrained_weights+'/'+checkpoint_name).expect_partial()
    freeze_all(model.get_layer('yolo_darknet'))

    optimizer = tf.keras.optimizers.legacy.Adam(lr=LEARNING_RATE)
    loss = [YoloLoss(anchors[mask], classes=NUM_CLASSES) for mask in anchor_masks]
    model.compile(optimizer=optimizer, loss=loss)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint(
            filepath=trained_weights+'/trained_weights.tf',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
    ]

    import os
    os.mkdir(trained_weights)

    model.fit(train_dataset, epochs=NUM_EPOCHS, callbacks=callbacks, validation_data=val_dataset)    

components.create_component_from_func(
    train_model,
    output_component_file='./component-files-yaml/train_model_component.yaml',
    packages_to_install=['yolov3-minimal']
)
