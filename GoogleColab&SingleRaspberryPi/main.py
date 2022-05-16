import numpy as np

import tensorflow as tf
import mnist_setup

batch_size = 64

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / np.float32(255)
y_train = y_train.astype(np.int64)
x_test = x_test / np.float32(255)
y_test = y_test.astype(np.int64)
single_train_dataset = tf.data.Dataset.from_tensor_slices(
  (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
single_val_dataset = tf.data.Dataset.from_tensor_slices(
  (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
single_test_dataset = tf.data.Dataset.from_tensor_slices(
  (x_test, y_test)).shuffle(10000).repeat().batch(batch_size)

single_worker_model = mnist_setup.build_and_compile_cnn_model()
print("Fit model on training data")
history =  single_worker_model.fit(single_train_dataset,
                        #validation_data=single_val_dataset,
                        epochs=3,
                        steps_per_epoch=70,
                        #validation_steps=70,
                        verbose=1)

print("The loss values and metric values during training:")
print(history.history)

print("Evaluate on test data")
results = single_worker_model.evaluate(single_test_dataset,
                                       steps=70)
print("test loss, test acc:", results)