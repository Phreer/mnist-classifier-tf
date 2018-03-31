import model
import numpy as np
import tensorflow as tf
import configure
import math
def create_model(model_name, imgs, keep_prob):
    var_s = set(tf.get_collection(tf.GraphKeys.VARIABLES))
    mnist_model = model.NNModel(model_name, imgs/127.5-1)
    mnist_model.add_conv2d(64, mapsize=3, stddev_factor=2.)
    mnist_model.add_conv2d(64, mapsize=3, stddev_factor=2.)
    bypass = mnist_model.output
    mnist_model.add_prelu()
    mnist_model.add_bottleneck_residual_block(128, BN=False)
    mnist_model.add_prelu()
    mnist_model.add_conv2d(64, mapsize=1, stddev_factor=2.)
    mnist_model.add_sum(bypass)
    mnist_model.add_relu()
    mnist_model.add_pooling(2)
    mnist_model.add_flatten()
    mnist_model.add_dense(1024, stddev_factor=2.)
    mnist_model.add_relu()
    mnist_model.add_dropout(keep_prob)
    mnist_model.add_dense(10, stddev_factor=2.)
    var_s = set(tf.get_collection(tf.GraphKeys.VARIABLES)) - var_s
    return mnist_model, var_s

def prepare_data(TRAIN_IMGS_FILENAME, TRAIN_LABELS_FILENAME, TEST_IMGS_FILENAME, TEST_LABELS_FILENAME):
    
    test_imgs = np.load(TEST_IMGS_FILENAME)
    test_imgs = test_imgs.reshape([test_imgs.shape[0], 28, 28, 1]).astype(np.float32)
    
    test_labels_onehot = np.load(TEST_LABELS_FILENAME)
    test_labels_onehot = test_labels_onehot.astype(np.float32)

    imgs = np.load(TRAIN_IMGS_FILENAME)
    imgs = imgs.reshape([imgs.shape[0], 28, 28, 1]).astype(np.float32)
    
    labels_onehot = np.load(TRAIN_LABELS_FILENAME)
    labels_onehot = labels_onehot.astype(np.float32)
    
    return (imgs, labels_onehot, test_imgs, test_labels_onehot)
            
def data_augmentation(imgs_input):
    BATCH_SIZE = int(imgs_input.shape[0])
    imgs_augmented = tf.pad(imgs_input, [[0,0], [configure.PADDING, configure.PADDING], [configure.PADDING,configure.PADDING], [0,0]])
    imgs_augmented = tf.random_crop(imgs_augmented, [BATCH_SIZE, configure.WIDTH, configure.HEIGHT, configure.CHANNELS])
    imgs_augmented = tf.contrib.image.rotate(imgs_augmented, np.random.choice(list(range(15)), 1)*math.pi/180.,\
                                                interpolation="BILINEAR")
    imgs_augmented = imgs_augmented + tf.truncated_normal([BATCH_SIZE, configure.WIDTH, configure.HEIGHT, configure.CHANNELS], mean=0, stddev=5)
    return imgs_augmented