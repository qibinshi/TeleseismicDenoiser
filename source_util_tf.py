import tensorflow as tf

# Define new loss with penalty on cheating
def custom_mse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true-y_pred))
    penalty = tf.reduce_mean(tf.square(tf.math.reduce_std(y_true, axis=0)
                                  - tf.math.reduce_std(y_pred, axis=0)))
    return mse + penalty

def custom_mae(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true-y_pred))
    penalty = tf.reduce_mean(tf.abs(tf.math.reduce_std(y_true, axis=0)
                                  - tf.math.reduce_std(y_pred, axis=0)))
    return mae + penalty