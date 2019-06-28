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
