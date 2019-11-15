import tensorflow as tf


def get_grid_coordinates(pred):
    """ Return a tensor containing the top left grid coordinate of each cell given a
    prediction tensor.

    :param pred: The tensor containing the predictions.
        - Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 5 + n_classes)
    :return:
        - grid: A tensor where each element represents the coordinates of a grid cell.
        - grid_size: A tensor with the size of the grid.
    """
    grid_size = pred.shape[1:3]
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0])) # Needs to be inverted.
    grid = tf.stack(grid, axis=-1)
    grid = tf.expand_dims(grid, axis=2)

    return tf.cast(grid, tf.float32), tf.convert_to_tensor([grid_size[1], grid_size[0]], dtype=tf.float32)


def split_outputs(output):
    """ Split the prediction tensor into it's separate components.

    :param pred: Tensor containing the predictions.
        - Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 5 + n_classes)
    :return:
        - box_xy: Tensor containing the XY predictions
            Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 2)
        - box_wh: Tensor containing the Width and Height predictions
            Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 2)
        - objectness: Tensor containing the Objectness confidence score
            Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 1)
        - class_probs: Tensor containing the probability of an object belonging to a class
            Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, n_classes)
    """
    n_classes = output.shape[4] - 5
    box_xy, box_wh, objectness, class_probs = tf.split(output, (2, 2, 1, n_classes), axis=4)
    return box_xy, box_wh, objectness, class_probs


def yolo_boxes(pred, anchors):
    """ Convert model predictions into Yolo outputs

    :param pred: Tensor containing the predictions.
        - Shape: (batch_size, grid_size_1, grid_size_2, n_anchors, 5 + n_classes)
    :param anchors: Tensor containing the Yolo Anchors
        - Shape: (n_anchors, 2)
    :return:
    """
    box_xy, box_wh, objectness, class_probs = split_outputs(pred)

    # Transforming the predictions to constrain the domain
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # Storing these values for later use in the loss function
    pred_box = tf.concat((box_xy, box_wh), axis=4)

    # Converting from grid cell coordinates into whole grid coordinates
    grid, grid_size = get_grid_coordinates(pred)
    box_xy = (box_xy + grid) / grid_size
    box_wh = tf.exp(box_wh) * anchors

    # Converting from (X,Y,W,H) coordinates to (TopLeft-XY, BottomRight-XY) coordinates
    box_tl = box_xy - box_wh / 2
    box_br = box_xy + box_wh / 2
    bbox = tf.concat([box_tl, box_br], axis=4)

    return bbox, objectness, class_probs, pred_box


def reverse_yolo_boxes(true_xy, true_wh, anchors):
    grid, grid_size = get_grid_coordinates(true_xy)

    true_xy = true_xy * grid_size - grid
    true_wh = tf.math.log(true_wh, anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh) # TODO: Why?

    return true_xy, true_wh



def YoloLoss(anchors, n_classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # Transforming the predicted outputs
        pred_bbox, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors)

        # Transforming the true outputs
        true_tl, true_br, true_obj, true_class_idx = split_outputs(y_true)
        true_box = tf.concat((true_tl, true_br), axis = 4)
        true_xy = (true_tl + true_br) / 2
        true_wh = (true_br - true_tl) / 2

        # Give smaller weights to small boxes TODO: Should we keep this?
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1] # This calculate bbox area

        # Reversing the Yolo-Box calculation
        true_xy, true_wh = reverse_yolo_boxes(true_xy, true_wh, anchors)

        #Calculating masks
        obj_mask = tf.cast(tf.squeeze(true_obj, axis=4), tf.bool)
        true_box_flat = tf.boolean_mask(true_box, obj_mask)
        best_iou = tf.reduce_max()

