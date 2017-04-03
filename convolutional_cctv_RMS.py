# /*******************************************************
#   Copyright (C) 2015-2017 Yifan Xie <yxyxyxyxyx@gmail.com>
#
#   This file is part of the proejct "imagesecurity",
#   and is written to be exploit within the scope of the aforementioned proejct
#
#   This code can not be copied and/or distributed without the expressed
#   permission of Yifan Xie
#  *******************************************************/

__author__ = 'Yifan Xie'

"""
This code is an adaptation from the convoluntional network tutorial from deeplearning.net.
It is an simplified version of the "LeNet" approach, details are described as below:

This implementation simplifies the model in the following ways:
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import numpy
import cPickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from random import randrange
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


best_params=[]
srng = RandomStreams()
import matplotlib.pyplot as plt


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def create_shared_dataset(dataset):
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    train_set, test_set=dataset
    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval


###########################################################################################################
# Impelmentation of RMSprop, this is a gradient descent algorithm that deciding the learning rate based
# on previous gradience.
def RMSprop_fixrate(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def RMSprop(cost, params, lr, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


# def evaluate_lenet5(datasets, imgh, imgw, nclass, lr_threshold, L1_reg=0.00, L2_reg=0.00, n_epochs=150, n_hidden=500,
#                     nkerns=[40, 70], batch_size=500):

def evaluate_lenet5(datasets, imgh, imgw, nclass, lr_threshold, L1_reg=0.00, L2_reg=0.00, n_epochs=150, n_hidden=500,
                    nkerns=[20, 50], batch_size=500):

    """
    :rtype : object
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nk+++++++++++++++++++++++++++++++++erns: number of kernels on each layer
    """
    global best_params
    filter_width=8

    rng = numpy.random.RandomState(23455)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x=T.tensor4('x')
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 54 * 36)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (54, 36) is the re-sized size of the cctv images.
    layer0_input = x.reshape((batch_size, 3, imgh, imgw))


    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (60-5+1 , 40-5+1) = (56, 36)
    # maxpooling reduces this further to (56/2, 36/2) = (28, 18)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 28, 18)
    #     image_shape=(batch_size, 3, 60, 40),

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imgh, imgw),
        filter_shape=(nkerns[0], 3, filter_width, filter_width),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (28-5+1, 18-5+1) = (24, 14)
    # maxpooling reduces this further to (24/2, 14/2) = (12, 7)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 12, 7)
    #     image_shape=(batch_size, nkerns[0], 28, 18),

    lh1=(imgh-filter_width+1)/2
    lw1=(imgw-filter_width+1)/2


    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], lh1, lw1),
        filter_shape=(nkerns[1], nkerns[0], filter_width, filter_width),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 12 * 7),
    # or (500, 50 * 12 * 7) = (500, 3360) with the default values.
    lh2=(lh1-filter_width+1)/2
    lw2=(lw1-filter_width+1)/2

    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * lh2 * lw2,
        n_out=n_hidden,
        activation=T.tanh
    )


    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=nclass)
    # noisy_layer3 = LogisticRegression(input=dropout(layer2.output,0.5), n_in=500, n_out=nclass)

    ### Regularization
    L1=(abs(layer0.W).sum()
        +abs(layer1.W).sum()
        +abs(layer2.W).sum()
        +abs(layer3.W).sum())

    L2_sqr=((layer0.W**2).sum()
        +(layer1.W**2).sum()
        +(layer2.W**2).sum()
        +(layer3.W**2).sum())


    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)+L1_reg*L1+L2_reg*L2_sqr

    # create a function to compute the mistakes that are made by the model
    # the following code is modified to suit with the small test set size
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]/255.0,
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params


    # theano expression to decay the learning rate across epoch
    current_rate=theano.tensor.fscalar('current_rate')

    updates = RMSprop(cost, params, current_rate)
    # updates = RMSprop_fixrate(cost, params, 0.001)


    train_model = theano.function(
        [index, current_rate],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]/255.0,
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50  # look at least at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_test_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False
    test_error=[]
    learning_rate=0.001
    while (epoch < n_epochs):
        epoch = epoch + 1
        # print "learning rate is %f" %learning_rate

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index, numpy.float32(learning_rate))
            # cost_ij = train_model(minibatch_index)

            if (iter + 1) % test_frequency == 0:
                # compute zero-one loss on validation set
                test_losses = [test_model(i) for i
                                     in xrange(n_test_batches)]
                this_test_loss = numpy.mean(test_losses)
                test_error.append(this_test_loss)
                print('epoch %i, minibatch %i/%i, test error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_test_loss * 100.))

                # optimizing learning rate for race classification
                if this_test_loss <lr_threshold and learning_rate>0.0001:
                    learning_rate=0.0001
                    # updates = RMSprop_fixrate(cost, params, 0.0001)
                    print "learning rate is now %0.0001"

                # if we got the best test score until now
                if this_test_loss < best_test_loss:

                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_test_loss = this_test_loss
                    best_iter = iter
                    best_params=layer3.params + layer2.params + layer1.params + layer0.params


    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_test_loss * 100., best_iter + 1, best_test_loss * 100.))
    print 'The code ran for %.2fm' %((end_time - start_time) / 60.)
    return best_params, test_error



###########################################################################################################
# allowing loading data from pickled file
def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data

###########################################################################################################
# allowing saving data into a pickle file
def pickle_data(path, data):
    file=path
    save_file=open(file, 'wb')
    cPickle.dump(data, save_file, -1)
    save_file.close()



if __name__ == '__main__':
    # Coding in this session is for development and testing purpose, and the comment is also for such purpose.

    # EC2 Setting
    # folder = os.path.dirname(__file__)
    # pickle_file=folder+"/home/ubuntu/pickle_data/image_secure_data.pkl"


    # Windows Setting
    folder="c:/users/xie/playground/cctv classification"
    pickle_file=folder+"/pickle_data/image_secure_data_54x36.pkl"

    data=load_data(pickle_file)
    img_list=data[0]
    gender_y=data[3]
    age_y=data[4]
    race_y=data[5]



    # sss=StratifiedShuffleSplit(gender_y, 1, test_size=0.25, random_state=0)
    sss=StratifiedShuffleSplit(race_y, 1, test_size=0.25, random_state=0)
    # # sss=StratifiedShuffleSplit(age_y[age_y!=5], 1, test_size=0.25, random_state=0)
    #
    #
    for train_index, test_index in sss:
        train_index_list=train_index
        test_index_list=test_index
        train_x, test_x=img_list[train_index], img_list[test_index]

        # perform split base on race shuffle, but apply the same split on gender and age
        train_ry, test_ry=race_y[train_index], race_y[test_index]
        train_gy, test_gy=gender_y[train_index], gender_y[test_index]

        # train_y, test_y=age_y[train_index], age_y[test_index]


        # set up dataset for pickle
        # train_set=[train_x, train_ry, train_gy]
        # test_set=[test_x,test_ry, test_gy]

        # set up dataset for race classification
        train_set=[train_x, train_ry]
        test_set=[test_x, test_ry]

        # set up dataset gender classification
        # train_set=[train_x, train_gy]
        # test_set=[test_x, test_gy]


        shuffled_dataset=[train_set, test_set]




        shared_dataset=create_shared_dataset(shuffled_dataset)

        # optimize for race classification: 0.25
        # optimize for gender classification: 0.28
        params, test_error=evaluate_lenet5(shared_dataset, 54, 36, 4, lr_threshold=0.25)

    #   Under unix, need to run as the following
    #   params, test_error=evaluate_lenet5(shared_dataset, 54, 36, 6)
    # #     plt.plot(test_error)

        model_file=folder+"/model/latest.pkl"
        pickle_data(model_file, params)


