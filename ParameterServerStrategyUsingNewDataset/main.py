import os
import tensorflow as tf
import json

def dataset_fn(input_context):
  global_batch_size = 1
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat()
  dataset = dataset.shard(
      input_context.num_input_pipelines, input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)
  return dataset

def build_and_compile_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    model.compile(tf.keras.optimizers.SGD(), loss="mse")
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cluster_spec = {
        "worker": ["192.168.0.30:4000",
                   "192.168.0.31:4000",
                   "192.168.0.32:4000"
        ],
        "ps": [ "192.168.0.164:55518",
                "192.168.0.61:22"
       ],
        "chief": ["192.168.0.25:55513"]
}

os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "ps", "index": 0}
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

with strategy.scope():
    multi_worker_model = build_and_compile_model()

multi_worker_model.fit(tf.keras.utils.experimental.DatasetCreator(
    dataset_fn), epochs=10, steps_per_epoch=10)