__author__ = 'xie'

import cPickle
import os
import gzip
import theano.tensor as T
import theano
import numpy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        print new_path
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def mlp_prediction(input, label, params):
    #cal mlp_prediction to check if the prediction is correct, i.e. equals to the label
    # mlp_prediction(input, label, params)
    hw, hb, lw, lb=params
    activation=T.tanh
    sample_x=T.matrix('sample_x')
    # lin_output =theano.function([sample_x], T.dot(sample_x, hw) + hb)
    # hidden_output=theano.function([], activation[lin_output])

    lin_output =T.dot(sample_x, hw) + hb
    hidden_output=activation(lin_output)
    logistic_ouput=T.argmax(T.nnet.softmax(T.dot(hidden_output, lw) + lb), axis=1)

    # sample_hidden_output=theano.function([sample_x], hidden_output)
    sample_logistic_ouput=theano.function([sample_x], logistic_ouput)

    pred_output=sample_logistic_ouput(input)

    print 'test label is %i' %label
    print 'predict label is %i' %pred_output



def LeNet_prediction(input, label, params):
    # Extract W and b value from each layer from params
    layer3_W=params[0]
    layer3_b=params[1]
    layer2_W=params[2]
    layer2_b=params[3]
    layer1_W=params[4]
    layer1_b=params[5]
    layer0_W=params[6]
    layer0_b=params[7]

   # setting necessary parameters
    poolsize=(2,2)
    layer0_img_shape=(1,1,28,28)
    layer0_filter_shape=(20,1,5,5)
    layer1_img_shape=(1,20,12,12)
    layer1_filter_shape=(50,20, 5,5)

    conv_input=T.dtensor4('conv_in')
    input=input.reshape(1,1,28,28)


    # layer0 symbolic expressions
    layer0_conv=conv.conv2d(input=conv_input,
        filters=numpy.float64(layer0_W.get_value()),
        filter_shape=layer0_filter_shape,
        image_shape=layer0_img_shape)
    layer0_pool=downsample.max_pool_2d(input=layer0_conv, ds=poolsize, ignore_border=True)
    layer0_output=T.tanh(layer0_pool + layer0_b.dimshuffle('x',0,'x', 'x'))

    # layer1 symbolic expressions
    layer1_conv=conv.conv2d(input=layer0_output,
        filters=numpy.float64(layer1_W.get_value()),
        filter_shape=layer1_filter_shape,
        image_shape=layer1_img_shape)
    layer1_pool=downsample.max_pool_2d(input=layer1_conv, ds=poolsize, ignore_border=True)
    layer1_output=T.tanh(layer1_pool + layer1_b.dimshuffle('x',0,'x', 'x'))

    # layer2 (hidden layer) expression
    layer2_input=layer1_output.flatten(2)
    layer2_lin_output =T.dot(layer2_input, layer2_W) + layer2_b
    layer2_output=T.tanh(layer2_lin_output)

    # layer3 (logistic layer) symbolic expression
    layer3_output=T.argmax(T.nnet.softmax(T.dot(layer2_output, layer3_W) + layer3_b), axis=1)

    # LeNet testing fuction on each level
    layer0_computation=theano.function([conv_input], layer0_output)
    layer1_computation=theano.function([conv_input], layer1_output)
    layer2_flatten=theano.function([conv_input], layer2_input)
    layer2_computation=theano.function([conv_input], layer2_output)
    layer3_computation=theano.function([conv_input], layer3_output)
    pred_output=layer3_computation(input)

    print 'test label is %i' %label
    print 'predict label is %i' %pred_output

########################################################
if __name__ == '__main__':
    model_path='./model/pre_train_sda.dat'
    load_file=open(model_path,'rb')
    params=cPickle.load(load_file)


    # dataset='mnist.pkl.gz'
    # datasets = load_data(dataset)
    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    #
    # # take one particular sample from the test data, and extract the value
    # test_label_index=T.iscalar('test_label_index')
    # test_label=theano.function([test_label_index], test_set_y[test_label_index])
    #
    # index=20
    # label=test_label(index)
    # input=test_set_x.get_value()[index,:].reshape(1,784)




    # LeNet_prediction(input,label, params)



    # mlp_prediction(input, label, params)




