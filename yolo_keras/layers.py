import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, LeakyReLU, Input, Add
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def DarknetConv(x, filters, size, strides=1, batch_norm=True, block_name="DarknetConv"):
    with K.name_scope(block_name):
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size,
                   strides=strides, padding=padding,
                   use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x


def DarknetResidual(x, filters, block_name="DarknetResidual"):
    with K.name_scope(block_name):
        prev = x
        x = DarknetConv(x, filters // 2, 1)
        x = DarknetConv(x, filters, 3)
        x = Add()([prev, x])
        return x


def DarknetBlock(input, filters, blocks, block_name="DarknetBlock"):
    with K.name_scope(block_name):
        x = DarknetConv(input, filters, 3, strides=2)
        for _ in range(blocks):
            x = DarknetResidual(x, filters)
        return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return Model(inputs, (x_36, x_61, x), name=name)

model = Darknet()

print(model.summary())

