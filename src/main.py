import tensorflow as tf
import tensorflow_addons as tfa

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    # Reshape into "channels last" setup.
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), data_format="channels_last"),
    # Groupnorm Layer
    tfa.layers.GroupNormalization(groups=5, axis=3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


