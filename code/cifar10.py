# coding:utf-8

import os
import tensorflow as tf

CATEGORY_NUM = 10
IMAGE_SHAPE = (32, 32, 3)
EXAMPLE_SPEC = {
    'image': tf.FixedLenFeature(IMAGE_SHAPE, tf.float32),
    'label': tf.FixedLenFeature([]         , tf.int64),
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
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_SHAPE[0]+8, IMAGE_SHAPE[1]+8)
    image = tf.random_crop(image, IMAGE_SHAPE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    label = tf.one_hot(parsed['label'], CATEGORY_NUM)
    return image, label

def _input_fn(tfrecord, params, training):
    batch_size       = params.get('batch_size', 512)
    buffer_size      = params.get('shuffle_buffer_size', 1024)
    compression_type = params.get('tfrecord_compression_type', 'NONE')
    dataset = (tf.data.TFRecordDataset(tfrecord, compression_type=compression_type)
        .map(_parse_proto)
        .shuffle(buffer_size))
    if training:
        dataset = dataset.repeat()
    images, labels = (dataset
        .batch(batch_size)
        .make_one_shot_iterator()
        .get_next())
    return {'images': images}, labels

def serving_input_fn(params):
    features = {
        'images': tf.placeholder(tf.float32, (None,) + IMAGE_SHAPE)
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

def model_fn(features, labels, mode, params):
    images = features['images']
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_or_create_global_step()
    
    # ハイパーパラメータの取得
    l2_regularizer_scale = params.get('l2_regularizer_scale', 1e-6)
    point_multiplier     = params.get('point_multiplier', 1)
    depth_multiplier     = params.get('depth_multiplier', 1)
    learning_rate        = params.get('learning_rate', 0.5)
    dropout_rate         = params.get('dropout_rate', 0.5)
    
    with tf.variable_scope('model',
        initializer=tf.glorot_uniform_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(l2_regularizer_scale)):
    
        # 畳み込みのブロック
        def conv_block(x, filters, strides, name=''):
            with tf.variable_scope('conv_block_' + name):
                x = tf.layers.conv2d(x, int(filters * point_multiplier),
                    3, strides, 'same', use_bias=False)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu(x)
            return x

        # 分離された畳み込みのブロック
        def sepconv_block(x, filters, strides, name=''):
            with tf.variable_scope('sepconv_block_'  + name):
                x = tf.layers.separable_conv2d(x, int(filters * point_multiplier),
                    3, strides, 'same', use_bias=False,
                    depth_multiplier=depth_multiplier)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu(x)
            return x

        # Google MobileNet
        x = images
        with tf.variable_scope('mobilenet'):
            x =    conv_block(x, 32  , 2, '32-2')
            x = sepconv_block(x, 64  , 1, '64-1')
            x = sepconv_block(x, 128 , 2, '128-2')
            x = sepconv_block(x, 128 , 1, '128-1')
            x = sepconv_block(x, 256 , 2, '256-2')
            x = sepconv_block(x, 256 , 1, '256-1')
            x = sepconv_block(x, 512 , 2, '512-2')
            x = sepconv_block(x, 512 , 1, '512-1-0')
            x = sepconv_block(x, 512 , 1, '512-1-1')
            x = sepconv_block(x, 512 , 1, '512-1-2')
            x = sepconv_block(x, 512 , 1, '512-1-3')
            x = sepconv_block(x, 512 , 1, '512-1-4')
            x = sepconv_block(x, 1024, 2, '1024-2')
            x = sepconv_block(x, 1024, 1, '1024-1')

        # 読み出し層
        with tf.variable_scope('readout_layer'):
            x = tf.reduce_mean(x, [1, 2], keep_dims=True, name='global_average_pooling')
            x = tf.layers.dropout(x, rate=dropout_rate,
                                  training=is_training, name='dropout')
            x = tf.layers.conv2d(x, CATEGORY_NUM, 1, name='readout')
            x = tf.squeeze(x, [1,2], name='squeeze')

        # 出力
        logits = tf.identity(x, name='logits')
        probabilities = tf.nn.softmax(logits, name='probabilities')
        
        # 予測したラベルのインデクス
        classes = tf.argmax(probabilities, axis=1, name='classes')

        predictions = {
            'classes': classes,
            'probabilities': probabilities,
        }
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions),
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

        # 誤差
        with tf.variable_scope('losses'):
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            regularization_loss = tf.losses.get_regularization_loss()
            total_loss = tf.losses.get_total_loss(name='total')

        # 最適化
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('optimization'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    fit = optimizer.minimize(total_loss, global_step)
        else:
            fit = None

        # 計測
        with tf.variable_scope('metrics'):
            metrics = {
                'accuracy': tf.metrics.accuracy(
                    tf.argmax(labels, axis=1),
                    predictions['classes'])
            }

        # TensorBoardに表示する情報(サマリ)
        with tf.variable_scope('summary'):
            tf.summary.image('images', images, max_outputs=6)
            tf.summary.scalar('regularization_loss', regularization_loss, family='losses')
            tf.summary.scalar('cross_entropy', cross_entropy, family='losses')
            tf.summary.scalar('total_loss', total_loss, family='losses')
            tf.summary.scalar('train_accuracy', metrics['accuracy'][1], family='metrics')

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=fit,
        eval_metric_ops=metrics)
