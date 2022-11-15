import tensorflow as tf



def pixel_shuffle(target, upscale=4):
  I = target
  upscale = upscale
  bsize, a, b, c = I.shape

  r = int(c / upscale)
  X = tf.reshape(I, (-1, a, b, r, r))
  X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
  X = tf.concat([tf.squeeze(x, 1) for x in X], 2)  # bsize, b, a*r, r
  X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
  X = tf.concat([tf.squeeze(x, 1) for x in X], 2)  # bsize, a*r, b*r
  return tf.reshape(X, (-1, a * r, b * r, 1))
