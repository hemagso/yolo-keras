import pytest
from yolo_keras.loss import get_grid_coordinates, split_outputs
import tensorflow as tf
import numpy as np


def test_get_grid_coordinates():
    n_classes = 5
    batch_size = 8
    grid_size = 2
    n_anchors = 9
    pred = np.zeros((batch_size, grid_size, grid_size, n_anchors, 5 + n_classes))
    pred = tf.convert_to_tensor(pred)
    grid, size = get_grid_coordinates(pred)
    correct = tf.convert_to_tensor([
        [
            [[0, 0]],
            [[1, 0]]
        ],
        [
            [[0, 1]],
            [[1, 1]]
        ]
    ], dtype=tf.float32)
    assert grid.shape == correct.shape
    tf.assert_equal(grid, correct)
    tf.assert_equal(size, tf.convert_to_tensor([grid_size, grid_size], dtype=tf.float32))


def test_split_predictions():
    n_classes = 5
    batch_size = 8
    grid_size = 13
    n_anchors = 9
    array = np.zeros((batch_size, grid_size, grid_size, n_anchors, 5 + n_classes))
    array[..., 0:2] = 1
    array[..., 2:4] = 2
    array[..., 4] = 3
    array[..., 5:5+n_classes] = 4
    pred = tf.convert_to_tensor(array, dtype=tf.float32)

    box_xy, box_wh, objectness, class_probs = split_outputs(pred)

    box_xy_correct_shape = (batch_size, grid_size, grid_size, n_anchors, 2)
    box_xy_correct = tf.convert_to_tensor(np.ones(box_xy_correct_shape), dtype=tf.float32)

    box_wh_correct_shape = (batch_size, grid_size, grid_size, n_anchors, 2)
    box_wh_correct = tf.convert_to_tensor(2 * np.ones(box_wh_correct_shape), dtype=tf.float32)

    objectness_correct_shape = (batch_size, grid_size, grid_size, n_anchors, 1)
    objectness_correct = tf.convert_to_tensor(3 * np.ones(objectness_correct_shape), dtype=tf.float32)

    class_probs_correct_shape = (batch_size, grid_size, grid_size, n_anchors, n_classes)
    class_probs_correct = tf.convert_to_tensor(4 * np.ones(class_probs_correct_shape), dtype=tf.float32)

    tf.assert_equal(box_xy_correct, box_xy)
    tf.assert_equal(box_wh_correct, box_wh)
    tf.assert_equal(objectness_correct, objectness)
    tf.assert_equal(class_probs_correct, class_probs)


test_get_grid_coordinates()
test_split_predictions()