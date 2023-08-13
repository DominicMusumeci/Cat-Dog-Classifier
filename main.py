import tensorflow as tf


graph = tf.Graph()
with graph.as_default():
    x = tf.constant(5, name="x_value")
    y = tf.constant(2, name="y_value")
    sum = tf.add(x, y, name="xy_sum")

    with tf.compat.v1.Session() as session:
        print("Result: ", sum.eval())