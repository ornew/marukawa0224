# coding:utf-8
# 現在、SageMakerではTensorFlowの実行はPython 2.7のみのサポートとなっているので注意

import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Mode

EXAMPLE_SPEC = {
    'image': tf.FixedLenFeature([28 * 28], tf.float32),
    'label': tf.FixedLenFeature([]       , tf.int64),
}

def train_input_fn(data_dir, params):
    tfrecord = os.path.join(data_dir, 'test.tfr')
    return _input_fn(tfrecord, params, training=True)
    
def eval_input_fn(data_dir, params):
    tfrecord = os.path.join(data_dir, 'test.tfr')
    return _input_fn(tfrecord, params, training=False)

def _parse_proto(proto):
    parsed = tf.parse_single_example(proto, EXAMPLE_SPEC)
    image = parsed['image']
    label = tf.one_hot(parsed['label'], 10)
    return image, label

def _input_fn(tfrecord, params, training):
    dataset = (tf.data.TFRecordDataset(tfrecord, compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        .map(_parse_proto)
        .shuffle(1000))
    # 学習のときは無限に生成する
    dataset = dataset.repeat() if training else dataset
    images, labels = (dataset
        .batch(params.get('batch_size', 512))
        .make_one_shot_iterator()
        .get_next())
    return {'images': images}, labels

def serving_input_fn(hparams):
    features = {
        'images': tf.placeholder(tf.float32, [None, 784])
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

def model_fn(features, labels, mode, params):    
    x = features['images']
    
    W = tf.get_variable('W', [784,10])
    b = tf.get_variable('b', [10])
    
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_indices = tf.argmax(input=y, axis=1) # 予測したラベルのインデクス
    
    predictions = {
        'classes': y_indices,
        'probabilities': y,
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions),
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    learning_rate = params.get('learning_rate', 0.5)
    global_step = tf.train.get_or_create_global_step()
    
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(y), reduction_indices=[1]))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    fit = optimizer.minimize(cross_entropy, global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=cross_entropy,
        train_op=fit,
        export_outputs=export_outputs)
