import tensorflow as tf
import time
import numpy as np
import sys


class NNModel(object):
    """Wrapped neural networks"""

    def __init__(self, model_name, features):
        """Setup a new model with name.

        args:
            name: str, the unique identifier to specify a model
            features: tensor to be the input of the model
        """
        self._model_name = model_name
        self._layer_outputs = [features]
        self.batch_size = int(features.shape[0])

    # Output utility functions
    @property
    def output(self):
        return self._layer_outputs[-1]

    @property
    def model_name(self):
        return self._model_name

    @property
    def outputs(self):
        return self._layer_outputs

    @property
    def layer_num(self):
        return len(self._layer_outputs)

    @property
    def _output_unit_num(self):
        return int(self.output.shape[-1])

    def _get_layer_str(self, layer=None):
        """give each layer a name to specify them

        args:
            layer int start from 1(input layer)
        """
        assert layer is None or layer > 0, "layer start from 1"
        if layer is None:
            layer = len(self.outputs)
        return "%s_L%03d" % (self.model_name, layer + 1)

    # Initialier utility functions
    def _he_initialized_tensor(self, prev_units, num_units, stddev_factor=2., distribution="normal"):
        """truncted_normal with limit = sqrt(6 / (fan_in + fan_out))"""
        assert (distribution == "normal" or distribution == "uniform")
        if distribution == "normal":
            stddev = np.sqrt(stddev_factor / prev_units)
            intw = tf.truncated_normal([prev_units, num_units], mean=0, stddev=stddev)
        elif distribution == "uniform":
            stddev = np.sqrt(stddev_factor / (prev_units + num_units))

            intw = tf.uniform([prev_units, num_units], mean=0., stddev=stddev)
        return intw

    def _he_initialized_tensor_conv2d(self, prev_units, num_units, mapsize, stddev_factor=2., distribution="normal"):
        """truncted_normal with limit = sqrt(6 / (fan_in + fan_out))"""
        assert (distribution == "normal" or distribution == "uniform")
        if distribution == "normal":
            stddev = np.sqrt(stddev_factor / (prev_units * mapsize * mapsize))
        elif distribution == "uniform":
            stddev = np.sqrt(stddev_factor / ((prev_units + num_units) * mapsize * mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units], mean=0., stddev=stddev)

    def _glorot_initialized_tensor(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs
        """
        stddev = np.sqrt(stddev_factor / np.sqrt(prev_units * num_units))
        return tf.truncated_normal([prev_units, num_units],
                                   mean=0.0, stddev=stddev)

    def _glorot_initialized_tensor_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs

        args:
            mapsize: int, filter size
        """

        stddev = np.sqrt(stddev_factor / (np.sqrt(prev_units * num_units) * mapsize * mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                   mean=0.0, stddev=stddev)

    # Add layer functions
    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.
        See ArXiv 1502.03167v3 for details.
        """

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.contrib.layers.batch_norm(self.output, scale=scale)

        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.reshape(self.output, [self.batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.
        Uses Glorot 2010 initialization assuming linear activation.
        """

        assert len(self.output.shape) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._output_unit_num

            # Weight term
            initw = self._he_initialized_tensor(prev_units, num_units,
                                                stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)

            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            out = tf.matmul(self.output, weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._output_unit_num
            out = tf.nn.sigmoid(self.output)

        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            # this_input = tf.square(self.output)
            reduction_indices = -1
            # acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            # out = this_input / (acc+FLAGS.epsilon)
            out = tf.nn.softmax(self.output, reduction_indices)
            # out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")

        self.outputs.append(out)
        return self

    def add_dropout(self, keep_prob):
        """Add a dropout layer to the net"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.dropout(self.output, keep_prob, seed=int(time.time()))
        self.outputs.append(out)
        return self

    def add_pooling(self, ksize, method="max", stride=2):
        """Add a pooling layer with stride, pooling method can be either max or average"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            if method == "max":
                out = tf.nn.max_pool(self.output, [1, ksize, ksize, 1], [1, stride, stride, 1], 'SAME')
            elif method == "average":
                out = tf.nn.pool(self.output, [1, ksize, ksize, 1], 'MAX', 'SAME', \
                                 strides=[1, stride, stride, 1])
        self.outputs.append(out)

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.relu(self.output)

        self.outputs.append(out)
        return self

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.elu(self.output)

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            # t1  = .5 * (1 + leak)
            # t2  = .5 * (1 - leak)
            # out = t1 * self.get_output() + \
            #      t2 * tf.abs(self.get_output())
            out = tf.nn.leaky_relu(self.output, alpha)
        self.outputs.append(out)
        return self

    def add_prelu(self, share_alpha=True):
        """Adds a paramatic PReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            alphas_shape = [1] if share_alpha else [self.output.shape[-1]]
            alphas = tf.get_variable("alphas", initializer=np.ones(shape=alphas_shape, dtype=np.float32) * 0.25,
                                     dtype=tf.float32)
            out = tf.nn.relu(self.output) + tf.multiply(alphas, (self.output - tf.abs(self.output))) * 0.5
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=2.0):
        """Adds a 2D convolutional layer."""

        assert len(self.output.shape) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._output_unit_num

            # Weight term and convolution
            initw = self._he_initialized_tensor_conv2d(prev_units, num_units, \
                                                       mapsize, \
                                                       stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out = tf.nn.conv2d(self.output, weight,
                               strides=[1, stride, stride, 1],
                               padding='SAME')

            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)

        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=2.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.output.shape) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._output_unit_num

            # Weight term and convolution
            initw = self._he_initialized_tensor_conv2d(prev_units, num_units,
                                                       mapsize,
                                                       stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.output
            output_shape = [batch_size,
                            int(prev_output.shape[1]) * stride,
                            int(prev_output.shape[2]) * stride,
                            num_units]
            out = tf.nn.conv2d_transpose(self.output, weight,
                                         output_shape=output_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME')

            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)

        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, \
                           num_layers=2, stddev_factor=2., BN=True):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.output.shape) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.output.shape[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=stddev_factor)

        # bypass = self.output
        # how if i use tf.identity?
        bypass = self.output
        # Residual block
        for _ in range(num_layers):
            if BN: self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False, BN=True, stddev_factor=2.):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.output.shape) == 4, "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.output.shape[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            # bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=stddev_factor)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=stddev_factor)

        bypass = self.output

        # Bottleneck residual block
        if BN: self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units // 4, mapsize=1, stride=1, stddev_factor=stddev_factor)

        if BN: self.add_batch_norm()
        # self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units // 4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=stddev_factor)
        else:
            self.add_conv2d(num_units // 4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=stddev_factor)

        if BN: self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.output.shape
            print("%s %s" % (prev_shape, term.shape))
            assert prev_shape[1:] == term.shape[1:], "Can't sum terms with a different size"
            out = tf.add(self.output, term)

        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.output.shape
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.output, reduction_indices=reduction_indices)

        self.outputs.append(out)
        return self

    def add_upscale(self, scale=2):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.output.shape
            size = [scale * int(s) for s in prev_shape[1:3]]
            out = tf.image.resize_nearest_neighbor(self.output, size)

        self.outputs.append(out)
        return self

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.
        The variable must already exist."""

        scope = self._get_layer_str(layer)
        collection = self.graph.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope + '/' + name:
                return var

        return None

    def get_all_layer_variables(self, layer=None):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return self.graph.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

    def save_model(self, sess, saver, global_step):
        CHECKPOINTS_PATH = "checkpoints/"
        saver.save(sess, os.path.join(CHECKPOINTS_PATH, self._model_name), global_step=global_step, )
        print("[*]Checkpoints saved.")

    def restore(self, sess, saver, global_step=None):
        CHECKPOINTS_PATH = "checkpoints/"

        checkpoint_name = os.path.join(CHECKPOINTS_PATH, \
                                       self._model_name + "-" + str(
                                           global_step)) if global_step is not None else tf.train.latest_checkpoint(
            CHECKPOINTS_PATH)
        try:
            print("[.]Restoring from {checkpoint_name} ...".format(checkpoint_name=checkpoint_name))
            saver.restore(sess, checkpoint_name)
            print("[*]Checkpoint loaded.")
        except Exception as e:
            print("[!]Unable to restore checkpoint!")
            raise e
        return

    def sample_output(self, sess, iteration, hr_imgs, sample_num=None):

        SAMPLE_PATH = "samples2/"

        if sample_num is None: sample_num = 1
        output_imgs = self.output[0:sample_num, :, :, :]
        bicubic = tf.image.resize_bicubic(self.outputs[0][0:sample_num, :, :, :],
                                          (output_imgs.shape[1], output_imgs.shape[2]))
        output_imgs = tf.concat([output_imgs, bicubic, hr_imgs[0:sample_num, :, :, :]], 2)
        output_imgs = tf.concat([output_imgs[i] for i in range(sample_num)], 0)
        output_imgs = sess.run(output_imgs)
        filename = "sample_output_t{iteration}.png".format(iteration=iteration)
        output_imgs = (np.array(output_imgs) * 255).astype("uint8")
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)
        pillow.Image.fromarray(output_imgs, "RGB").save(os.path.join(SAMPLE_PATH, filename))


class VGG:
    LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = scipy.io.loadmat(data_path)
        mean = self.data['normalization'][0][0][0]
        self.mean_pixel = np.mean(mean, axis=(0, 1))
        self.weights = self.data['layers'][0]

    def preprocess(self, image):
        return image - self.mean_pixel

    def unprocess(self, image):
        return image + self.mean_pixel

    def net(self, input_image):
        net = {}
        current_layer = input_image
        for i, name in enumerate(self.LAYERS):
            if _is_convolutional_layer(name):
                kernels, bias = self.weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current_layer = _conv_layer_from(current_layer, kernels, bias)
            elif _is_relu_layer(name):
                current_layer = tf.nn.relu(current_layer)
            elif _is_pooling_layer(name):
                current_layer = _pooling_layer_from(current_layer)
            net[name] = current_layer

        assert len(net) == len(self.LAYERS)
        return net


def _is_convolutional_layer(name):
    return name[:4] == 'conv'


def _is_relu_layer(name):
    return name[:4] == 'relu'


def _is_pooling_layer(name):
    return name[:4] == 'pool'


def _conv_layer_from(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pooling_layer_from(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
