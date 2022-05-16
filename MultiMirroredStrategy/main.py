import os
import json
import numpy as np

import tensorflow as tf
import mnist_setup

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
per_worker_batch_size = 64

## Manual cluster setup
os.environ["TF_CONFIG"] = json.dumps({
   "cluster": {
       "chief": ["192.168.0.25:55513"],
       "worker": ["192.168.0.30:4000", "192.168.0.31:4000", "192.168.0.32:4000"],
   },
  "task": {"type": "worker", "index": 2}
})

tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / np.float32(255)
y_train = y_train.astype(np.int64)
x_test = x_test / np.float32(255)
y_test = y_test.astype(np.int64)

multi_train_dataset = tf.data.Dataset.from_tensor_slices(
  (x_train, y_train)).shuffle(60000).repeat().batch(global_batch_size)
multi_val_dataset = tf.data.Dataset.from_tensor_slices(
  (x_train, y_train)).shuffle(60000).repeat().batch(global_batch_size)
multi_test_dataset = tf.data.Dataset.from_tensor_slices(
  (x_test, y_test)).shuffle(10000).repeat().batch(global_batch_size)

# Model building/compiling need to be within 'strategy.scope()'.
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

print("Fit model on training data")
history = multi_worker_model.fit(multi_train_dataset,
                                 #validation_data=multi_val_dataset,
                                 epochs=3,
                                 steps_per_epoch=70,
                                 #validation_steps=70,
                                 verbose=1)

print("The loss values and metric values during training:")
print(history.history)

print("Evaluate on test data")
results = multi_worker_model.evaluate(multi_test_dataset, steps=70)
print("test loss, test acc:", results)