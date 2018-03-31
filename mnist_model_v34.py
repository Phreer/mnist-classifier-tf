import tensorflow as tf
import numpy as np
import utils
import configure
import time
import tf_cnnvis


class mnist_v34():
    def __init__(self, name, graph, SAVE_DIR):
        self.restored = False
        self.graph = graph
        self.BATCH_SIZE = 1
        self.SAVE_DIR = SAVE_DIR
        with self.graph.as_default():
            with tf.name_scope("input"):
                self.imgs_input = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE, configure.WIDTH, configure.HEIGHT, 1],
                                            name="imgs")
                self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
            self.mnist_model, self.var_s = utils.create_model(name, self.imgs_input, self.keep_prob)
            self.sess = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver()
            print(self.mnist_model.outputs)

    def _restore(self):
        with self.graph.as_default():
            print("loading model...")
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.SAVE_DIR))
            self.restored = True

    def predict(self, img):
        if(len(img.shape)==3):
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        elif(len(img.shape)==2):
            img = img.reshape((1, img.shape[0], img.reshape[1], 1))
        with self.graph.as_default():
            if not self.restored: self._restore()
            print(img.shape, img.dtype)
            feed_dict = {self.imgs_input: img, self.keep_prob: 1.}
            pred = self.sess.run(self.mnist_model.output, feed_dict=feed_dict)
            print(pred)
            return pred
    def visualize_activation(self, img):
        if(len(img.shape)==3):
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        elif(len(img.shape)==2):
            img = img.reshape((1, img.shape[0], img.reshape[1], 1))
        with self.graph.as_default():
            if not self.restored: self._restore()
            print(img.shape, img.dtype)
            feed_dict = {self.keep_prob: 1.}
            tf_cnnvis.deepdream_visualization(self.sess, feed_dict, layer=self.mnist_model.output.name, classes=list(range(10)))

class mnist_model_emsembled:
    def __init__(self):
        self.graph1 = tf.Graph()
        self.graph2 = tf.Graph()
        self.graph3 = tf.Graph()
        self.SAVE_DIR1 = r"E:\mnist_model_v3_4_vis\save\save1"
        self.SAVE_DIR2 = r"E:\mnist_model_v3_4_vis\save\save2"
        self.SAVE_DIR3 = r"E:\mnist_model_v3_4_vis\save\save3"
        self.mnist1 = mnist_v34("mnist_classifier_v3_3_b1", self.graph1, self.SAVE_DIR1)
        self.mnist2 = mnist_v34("mnist_classifier_v3_3_b2", self.graph2, self.SAVE_DIR2)
        self.mnist3 = mnist_v34("mnist_classifier_v3_3_b3", self.graph3, self.SAVE_DIR3)

    def _restore(self):
        self.mnist1._restore()
        self.mnist2._restore()
        self.mnist3._restore()

    def predict(self, img):
        pred = self.mnist1.predict(img) + self.mnist2.predict(img) + self.mnist3.predict(img)
        print(pred)
        return pred
    def visualize_activation(self, img):
        self.mnist1.visualize_activation(img)


if __name__ == "__main__":
    test_imgs = np.load(r'E:\mnist_model_v3_4_vis\test_imgs_array.np')
    img = test_imgs[84].reshape([1, 28, 28, 1]).astype(np.float32)
    emsembled = mnist_model_emsembled()
    emsembled._restore()
    start = time.time()
    pred = emsembled.predict(img)
    print("Time consumed: ", time.time()-start)
    from matplotlib import pyplot
    emsembled.visualize_activation(img)
    pyplot.imshow(img.reshape(28, 28))
    pyplot.show()
