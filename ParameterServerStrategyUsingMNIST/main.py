import os
import tensorflow as tf
import json
import mnist_setup

per_worker_batch_size = 64

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cluster_spec = {
        "worker": ["192.168.0.30:4000",
                   "192.168.0.31:4000",
                   "192.168.0.32:4000"
        ],
        "ps": [ "192.168.0.33:4000",
                "192.168.0.34:4000"
       ],
        "chief": ["192.168.0.25:55513"]
}

os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "ps", "index": 1}
})

tf_config = json.loads(os.environ['TF_CONFIG'])

os.environ["GRPC_FAIL_FAST"] = "use_caller"
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if cluster_resolver.task_type in ("worker", "ps"):
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    server.join()

strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

INPUT = tf.keras.utils.experimental.DatasetCreator(mnist_setup.mnist_dataset)

with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

multi_worker_model.fit(INPUT, epochs=3, steps_per_epoch=70)