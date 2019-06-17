import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # convolution layers
      C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)

      #### Residual Network ####
      #filter_shape = [1, 128, 128, 64]
      # = create_variables(name='conv', shape=filter_shape)
      #conv = tf.nn.conv2d(C64, filter,
      #  strides=[1, 1, 1, 1], padding='SAME')

      C64_res_1 = ops.Rk(C64, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64_res_1')

      C64_res_2 = ops.Rk(C64_res_1, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64_res_2')

      C128_norm = ops.Ck(C64_res_2, 128, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C128_norm')

      C128_res_1 = ops.Rk(C128_norm, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128_res_1')            # (?, w/4, h/4, 128)

      C128_res_2 = ops.Rk(C128_res_1, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128_res_2')

      C256_norm = ops.Ck(C128_res_2, 256, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C256_norm')

      C256_res_1 = ops.Rk(C256_norm, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256_res_1')            # (?, w/8, h/8, 256)

      C256_res_2 = ops.Rk(C256_res_1, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256_res_2')            # (?, w/8, h/8, 256)

      C512_norm = ops.Ck(C256_res_2, 512, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C512_norm')

      C512_res_1 = ops.Rk(C512_norm, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512_res_1')            # (?, w/16, h/16, 512)

      C512_res_2 = ops.Rk(C512_res_1, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512_res_2')            # (?, w/16, h/16, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      output = ops.last_conv(C512_res_2, reuse=self.reuse,
          use_sigmoid=self.use_sigmoid, name='output')          # (?, w/16, h/16, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
