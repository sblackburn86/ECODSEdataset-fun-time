import tensorflow as tf

tf.enable_eager_execution()


class TestMLP(tf.keras.Model):
    """
    A fully connected neural network with 1 hidden layer.

    Attributes
    ----------
    model : tf.keras.Model
        sequential model made of 1 hidden layer with ReLU activation and 1 output layer.

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, hiddensize, outsize):
        """
        Class construction

        Parameters
        ----------
        hiddensize : int
            size of the hidden layer
        outsize : int
            size of the output layer
        """
        super(TestMLP, self).__init__(self)
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Reshape((256*256*3,)),
            tf.keras.layers.Dense(hiddensize, input_shape=(256 * 256 * 3,), activation=tf.nn.relu),
            tf.keras.layers.Dense(outsize, activation=tf.nn.sigmoid)
        ])

    def call(self, inputs):
        """
        forward propagation function of the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256 * 256 * 3)
            mini-batch of batchsize flatten images

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.model(inputs)


class SimpleCNN(tf.keras.Model):
    """
    A convolutional neural network with 1 conv layer, 1 maxpool, and 1 fully-connected output layer.

    Attributes
    ----------
    model : tf.keras.Model
        sequential model made of 1 conv layer, 1 maxpool layer and 1 dense layer

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, nkernel, kernelsize, outsize):
        """
        Class constructor

        Parameters
        ----------
        nkernel : int
            number of kernels in the convolutional layer
        kernelsize : int
            size of the kernels in the convolutional layer
        outsize : int
            number of dimensions of the output
        """
        super(SimpleCNN, self).__init__(self)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(nkernel, kernelsize, input_shape=(256, 256, 3,), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(outsize)
        ])

    def call(self, inputs):
        """
        Forward function for the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256, 256, 3)
            mini-batch of batchsize images of size 256 x 256 x 3

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.model(inputs)


class FuntimeResnet50(tf.keras.Model):
    """
    A retrainable resnet

    Attributes
    ----------
    base_model : tf.keras.applications.ResNet50
        an instance of a resnet50 model with weights initialized from imagenet
    prediction_layer: tf.keras.layers.Dense
        flatten the resnet output and a fully connected layers to calculate the prediction logits

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, outsize, train_resnet=False):
        """
        Class constructor

        Parameters
        ----------
        outsize : int
            number of dimensions of the output
        train_resnet : bool, optional
            if True, weights in the resnet50 are trainable;
            if False, weights are frozen;
            by default False;
        """
        super(FuntimeResnet50, self).__init__(self)
        self.base_model = tf.keras.applications.ResNet50(input_shape=(256, 256, 3),
                                                         include_top=False,
                                                         weights='imagenet')
        self.base_model.trainable = train_resnet
        self.prediction_layer = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(outsize)
        ])

    def call(self, inputs):
        """
        Forward pass for the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256, 256, 3)
            mini-batch of batchsize images of size 256 x 256 x 3

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.prediction_layer(self.base_model(inputs))


class PretrainedVGG16(tf.keras.Model):
    """
    A retrainable VGG16 net

    Attributes
    ----------
    base_model : tf.keras.applications.VGG16
        an instance of a VGG16 model with weights initialized from imagenet
    prediction_layer: tf.keras.layers.Dense
        flatten the resnet output and a fully connected layers to calculate the prediction logits

    Methods
    -------
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self, outsize, train_vgg=False, pooling=None):
        """
        Class constructor

        Parameters
        ----------
        outsize : int
           number of dimensions of the output
        train_vgg : bool, optional
            if True, weights in the vgg16 are trainable;
            if False, weights are frozen;
            by default False;
        pooling : string, optional
            type of pooling to apply after the VGG network
            None: no pooling is applied, the output is flatten
            'avg': average pooling
            'max' : max pooling
        """
        super(PretrainedVGG16, self).__init__(self)
        if pooling:
            assert pooling in ["avg", "max"], "Pooling should be None, avg, or max"

        self.base_model = tf.keras.applications.VGG16(input_shape=(256, 256, 3),
                                                      include_top=False,
                                                      weights='imagenet',
                                                      pooling=pooling)
        self.base_model.trainable = train_vgg
        if pooling:
            self.prediction_layer = tf.keras.layers.Dense(outsize)
        else:
            self.prediction_layer = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(outsize)
            ])

    def call(self, inputs):
        """
        Forward pass for the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256, 256, 3)
            mini-batch of batchsize images of size 256 x 256 x 3

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.prediction_layer(self.base_model(inputs))


class CustomVGG16(tf.keras.Model):
    """
    A customizable VGG16 net

    Attributes
    ----------
    conv_layers : tf.keras.Sequential
        a sequence of 5 convolution blocks based on the VGG16 architecture
    dense_layers: tf.keras.Sequential
        a sequence of:
            tf.keras.layers.Flatten() : flatten the output of the convolution blocks
            3 tf.keras.layers.Dense() : fully-connected layers to output the predictions

    Methods
    -------
    conv_block:
        creates a block of convolutional layers and a max_pooling layer
    call:
        inherit from keras
        Usage example: model(x)
    """

    def __init__(self,
                 filter1=64,
                 kernel_size1=3,
                 stride1=1,
                 pool1=2,
                 stride_pool1=2,
                 filter2=128,
                 kernel_size2=3,
                 stride2=1,
                 pool2=2,
                 stride_pool2=2,
                 filter3=256,
                 kernel_size3=3,
                 stride3=1,
                 pool3=2,
                 stride_pool3=2,
                 filter4=512,
                 kernel_size4=3,
                 stride4=1,
                 pool4=2,
                 stride_pool4=2,
                 filter5=512,
                 kernel_size5=3,
                 stride5=1,
                 pool5=2,
                 stride_pool5=2,
                 dense1=4096,
                 dense2=4096,
                 outsize=9):
        """
        Class constructor

        Parameters
        ----------
        filter1 : int, optional
            number of filters in the 1st convolutional block, by default 64
        kernel_size1 : int, optional
            kernel size in the 1st convolutional block, by default 3
        stride1: int, optional
            stride size in the 1st convolutional block, by default 1
        pool1 : int, optional
            pooling size in the 1st convolutional block, by default 2
        filter2 : int, optional
            number of filters in the 2nd convolutional block, by default 128
        kernel_size2 : int, optional
            kernel size in the 2nd convolutional block, by default 3
        pool2 : int, optional
            pooling size in the 2nd convolutional block,, by default 2
        filter3 : int, optional
            number of filters in the 3rd convolutional block, by default 256
        kernel_size3 : int, optional
            kernel size in the 3rd convolutional block, by default 3
        pool3 : int, optional
            pooling size in the 3rd convolutional block,, by default 2
        filter4 : int, optional
            number of filters in the 4th convolutional block, by default 512
        kernel_size4 : int, optional
            kernel size in the 4th convolutional block, by default 3
        pool4 : int, optional
            pooling size in the 4th convolutional block,, by default 2
        filter5 : int, optional
            number of filters in the 5th convolutional block, by default 512
        kernel_size5 : int, optional
            kernel size in the 5th convolutional block, by default 3
        pool5 : int, optional
            pooling size in the 5th convolutional block,, by default 2
        dense1 : int, optional
            number of units in the 1st dense layer, by default 4096
        dense2 : int, optional
            number of units in the 1st dense layer, by default 4096
        outsize : int, optional
            number of output classes, by default 9
        """

        super(CustomVGG16, self).__init__(self)

        # check size of the output first starting from 256x256 image
        # we are using same padding, so conv layers do not change the input size
        out1 = (256 - pool1) // stride_pool1 + 1
        out2 = (out1 - pool2) // stride_pool2 + 1
        out3 = (out2 - pool3) // stride_pool3 + 1
        out4 = (out3 - pool4) // stride_pool4 + 1
        out5 = (out4 - pool5) // stride_pool5 + 1
        assert out5 >= 1, "Convolution output has shape 0 or negative"

        self.conv_layers = tf.keras.Sequential([
            self.convblock(filter1, kernel_size1, stride1, pool1, stride_pool1, 2),
            self.convblock(filter2, kernel_size2, stride2, pool2, stride_pool2, 2),
            self.convblock(filter3, kernel_size3, stride3, pool3, stride_pool3, 2),
            self.convblock(filter4, kernel_size4, stride4, pool4, stride_pool4, 3),
            self.convblock(filter5, kernel_size5, stride5, pool5, stride_pool5, 3)
        ])

        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense1, activation="relu"),
            tf.keras.layers.Dense(dense2, activation="relu"),
            tf.keras.layers.Dense(outsize, activation=None)
        ])

    def convblock(self, filter, kernel, stride=1, pooling=2, pool_stride=2, blocksize=2):
        """
        Creates a convolution block in the VGG architecture

        Parameters
        ----------
        filter : int
            number of filters in the convolution layers
        kernel : int
            size of filters in the convolution layers
        stride: int, optional
            stride size in the convolution layers, by default 1
        pooling : int, optional
            size of max pooling, by default 2
        pool_stride : int, optional
            stride size in the max pool layer, by default 2
        blocksize : int, optional
            number of convolution blocks, by default 2
        """
        assert blocksize in [2, 3], "Number of conv layers should be 2 or 3"
        if blocksize == 2:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=(stride, stride), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=(stride, stride), padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(pool_size=(pooling, pooling), strides=(pool_stride, pool_stride))
            ])
        if blocksize == 3:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=(stride, stride), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=(stride, stride), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=(stride, stride), padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(pool_size=(pooling, pooling), strides=(pool_stride, pool_stride))
            ])

    def call(self, inputs):
        """
        Forward pass for the model

        Parameters
        ----------
        inputs : tf.tensor shape = (batchsize, 256, 256, 3)
            mini-batch of batchsize images of size 256 x 256 x 3

        Returns
        -------
        tf.tensor shape = (batchsize, outsize)
            logits prediction of the model for the outsize classes
        """
        return self.dense_layers(self.conv_layers(inputs))
