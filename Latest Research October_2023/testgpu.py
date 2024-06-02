import tensorflow as tf
gpu_available = tf.config.list_physical_devices('GPU')
print(gpu_available)