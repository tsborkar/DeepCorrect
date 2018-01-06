from keras.models import Model
from keras.layers import Flatten, Dense, Dropout,  Activation, \
    Input, merge, BatchNormalization, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AtrousConvolution2D
from keras.regularizers import l1, l2
import numpy as np

from convnetskeras.customlayers import crosschannelnormalization, \
    splittensor
import h5py
import Imagenet_datagen as IN_dat

import keras.backend as K

layer_names = ['conv_1','conv_2','conv_3','conv_4','conv_5']

layer_size101 = np.empty((5,3),int)
layer_size101[0,:] = [96,55,55]
layer_size101[1,:] = [256,27,27]
layer_size101[2,:] = [384,13,13]
layer_size101[3,:] = [384,13,13]
layer_size101[4,:] = [256,13,13]

# build baseline AlexNet
def AlexNetDNN(weights_path=None, heatmap=False,trainable=False, out_dim =1000):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(3,227,227))

    inp_file = h5py.File(weights_path)

    conv1_w = inp_file['conv_1/conv_1_W'][:]
    conv1_b = inp_file['conv_1/conv_1_b'][:]
    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',weights=[conv1_w,conv1_b],
                           name='conv_1',trainable=trainable)(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)

    conv2_w_t = inp_file['conv_2_1/conv_2_1_W'][:]
    conv2_w = np.empty((2,conv2_w_t.shape[0],conv2_w_t.shape[1],conv2_w_t.shape[2],conv2_w_t.shape[3]),np.float32)
    conv2_w[0,:] = conv2_w_t.copy()
    del conv2_w_t
    conv2_w[1,:] = inp_file['conv_2_2/conv_2_2_W'][:]

    conv2_b_t = inp_file['conv_2_1/conv_2_1_b'][:]
    conv2_b = np.empty((2,conv2_b_t.shape[0]),np.float32)
    conv2_b[0,:] = conv2_b_t.copy()
    del conv2_b_t
    conv2_b[1,:] = inp_file['conv_2_2/conv_2_2_b'][:]

    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1),weights=[conv2_w[i,:],conv2_b[i,:]],trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)


    conv3_w = inp_file['conv_3/conv_3_W'][:]
    conv3_b = inp_file['conv_3/conv_3_b'][:]
    conv_3 = Convolution2D(384,3,3,weights=[conv3_w,conv3_b],activation='relu',name='conv_3',trainable=trainable)(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)

    conv4_w_t = inp_file['conv_4_1/conv_4_1_W'][:]
    conv4_w = np.empty((2,conv4_w_t.shape[0],conv4_w_t.shape[1],conv4_w_t.shape[2],conv4_w_t.shape[3]),np.float32)
    conv4_w[0,:] = conv4_w_t.copy()
    del conv4_w_t
    conv4_w[1,:] = inp_file['conv_4_2/conv_4_2_W'][:]

    conv4_b_t = inp_file['conv_4_1/conv_4_1_b'][:]
    conv4_b = np.empty((2,conv4_b_t.shape[0]),np.float32)
    conv4_b[0,:] = conv4_b_t.copy()
    del conv4_b_t
    conv4_b[1,:] = inp_file['conv_4_2/conv_4_2_b'][:]


    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1),weights=[conv4_w[i,:],conv4_b[i,:]],trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)

    conv5_w_t = inp_file['conv_5_1/conv_5_1_W'][:]
    conv5_w = np.empty((2,conv5_w_t.shape[0],conv5_w_t.shape[1],conv5_w_t.shape[2],conv5_w_t.shape[3]),np.float32)
    conv5_w[0,:] = conv5_w_t.copy()
    del conv5_w_t
    conv5_w[1,:] = inp_file['conv_5_2/conv_5_2_W'][:]

    conv5_b_t = inp_file['conv_5_1/conv_5_1_b'][:]
    conv5_b = np.empty((2,conv5_b_t.shape[0]),np.float32)
    conv5_b[0,:] = conv5_b_t.copy()
    del conv5_b_t
    conv5_b[1,:] = inp_file['conv_5_2/conv_5_2_b'][:]


    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1),weights=[conv5_w[i,:],conv5_b[i,:]],trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1_w = inp_file['dense_1/dense_1_W'][:]
    dense_1_b = inp_file['dense_1/dense_1_b'][:]

    dense_2_w = inp_file['dense_2/dense_2_W'][:]
    dense_2_b = inp_file['dense_2/dense_2_b'][:]

    dense_3_w = inp_file['dense_3/dense_3_W'][:]
    dense_3_b = inp_file['dense_3/dense_3_b'][:]

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1',weights=[dense_1_w,dense_1_b],trainable=trainable)(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2',weights=[dense_2_w,dense_2_b],trainable=trainable)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    if out_dim ==1000:
        dense_3 = Dense(out_dim,name='dense_3', weights=[dense_3_w,dense_3_b],trainable=trainable)(dense_3)
    else:
        # change trainable from true to trainable
        dense_3 = Dense(out_dim,name='dense_3',trainable=True,init='he_normal')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    return model, layer_dict

# build AlexNet layers
def AlexNetDNN_layers(weights_path=None,layer_id=0,trainable=False, out_dim =1000):


    learning_rate_multiplier = 1.0
    tr_wts = np.load(weights_path)
    if layer_id == 0:
        inputs = Input(shape=(96, 55, 55))
        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(inputs)
        conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)

        conv2_w_t = tr_wts[2]
        conv2_w = np.empty((2, conv2_w_t.shape[0], conv2_w_t.shape[1], conv2_w_t.shape[2], conv2_w_t.shape[3]), np.float32)
        conv2_w[0, :] = conv2_w_t.copy()
        del conv2_w_t
        conv2_w[1, :] = tr_wts[4]

        conv2_b_t = tr_wts[3]
        conv2_b = np.empty((2, conv2_b_t.shape[0]), np.float32)
        conv2_b[0, :] = conv2_b_t.copy()
        del conv2_b_t
        conv2_b[1, :] = tr_wts[5]

        conv_2 = merge([ Convolution2D(128, 5, 5, activation="relu", name='conv_2_' + str(i + 1),weights=[conv2_w[i, :], conv2_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_2)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)

        conv3_w = tr_wts[6]
        conv3_b = tr_wts[7]
        conv_3 = Convolution2D(384, 3, 3, weights=[conv3_w, conv3_b], activation='relu', name='conv_3', trainable=trainable)(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)

        conv4_w_t = tr_wts[8]
        conv4_w = np.empty((2, conv4_w_t.shape[0], conv4_w_t.shape[1], conv4_w_t.shape[2], conv4_w_t.shape[3]), np.float32)
        conv4_w[0, :] = conv4_w_t.copy()
        del conv4_w_t
        conv4_w[1, :] = tr_wts[10]

        conv4_b_t = tr_wts[9]
        conv4_b = np.empty((2, conv4_b_t.shape[0]), np.float32)
        conv4_b[0, :] = conv4_b_t.copy()
        del conv4_b_t
        conv4_b[1, :] = tr_wts[11]

        conv_4 = merge([
                           Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1),
                                         weights=[conv4_w[i, :], conv4_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

        conv_5 = ZeroPadding2D((1, 1))(conv_4)

        conv5_w_t = tr_wts[12]
        conv5_w = np.empty((2, conv5_w_t.shape[0], conv5_w_t.shape[1], conv5_w_t.shape[2], conv5_w_t.shape[3]), np.float32)
        conv5_w[0, :] = conv5_w_t.copy()
        del conv5_w_t
        conv5_w[1, :] = tr_wts[14]

        conv5_b_t = tr_wts[13]
        conv5_b = np.empty((2, conv5_b_t.shape[0]), np.float32)
        conv5_b[0, :] = conv5_b_t.copy()
        del conv5_b_t
        conv5_b[1, :] = tr_wts[15]

        conv_5 = merge([
                           Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1),
                                         weights=[conv5_w[i, :], conv5_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

        dense_1_w = tr_wts[16]
        dense_1_b = tr_wts[17]

        dense_2_w = tr_wts[18]
        dense_2_b = tr_wts[19]

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=trainable)(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=trainable)(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20],tr_wts[21]],trainable=trainable, init='he_normal')(dense_3)
        prediction = Activation("softmax", name="softmax")(dense_3)

    elif layer_id == 1:
        inputs = Input(shape=(256, 27, 27))
        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(inputs)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)

        conv3_w = tr_wts[6]
        conv3_b = tr_wts[7]
        conv_3 = Convolution2D(384, 3, 3, weights=[conv3_w, conv3_b], activation='relu', name='conv_3',trainable=trainable)(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)

        conv4_w_t = tr_wts[8]
        conv4_w = np.empty((2, conv4_w_t.shape[0], conv4_w_t.shape[1], conv4_w_t.shape[2], conv4_w_t.shape[3]),
                           np.float32)
        conv4_w[0, :] = conv4_w_t.copy()
        del conv4_w_t
        conv4_w[1, :] = tr_wts[10]

        conv4_b_t = tr_wts[9]
        conv4_b = np.empty((2, conv4_b_t.shape[0]), np.float32)
        conv4_b[0, :] = conv4_b_t.copy()
        del conv4_b_t
        conv4_b[1, :] = tr_wts[11]

        conv_4 = merge([
                           Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1),
                                         weights=[conv4_w[i, :], conv4_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

        conv_5 = ZeroPadding2D((1, 1))(conv_4)

        conv5_w_t = tr_wts[12]
        conv5_w = np.empty((2, conv5_w_t.shape[0], conv5_w_t.shape[1], conv5_w_t.shape[2], conv5_w_t.shape[3]),
                           np.float32)
        conv5_w[0, :] = conv5_w_t.copy()
        del conv5_w_t
        conv5_w[1, :] = tr_wts[14]

        conv5_b_t = tr_wts[13]
        conv5_b = np.empty((2, conv5_b_t.shape[0]), np.float32)
        conv5_b[0, :] = conv5_b_t.copy()
        del conv5_b_t
        conv5_b[1, :] = tr_wts[15]

        conv_5 = merge([
                           Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1),
                                         weights=[conv5_w[i, :], conv5_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)


        dense_1_w = tr_wts[16]
        dense_1_b = tr_wts[17]

        dense_2_w = tr_wts[18]
        dense_2_b = tr_wts[19]

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=trainable)(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=trainable)(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20], tr_wts[21]], trainable=trainable,
                        init='he_normal')(dense_3)
        prediction = Activation("softmax", name="softmax")(dense_3)

    elif layer_id == 2:
        inputs = Input(shape=(384, 13, 13))
        conv_4 = ZeroPadding2D((1, 1))(inputs)

        conv4_w_t = tr_wts[8]
        conv4_w = np.empty((2, conv4_w_t.shape[0], conv4_w_t.shape[1], conv4_w_t.shape[2], conv4_w_t.shape[3]),
                           np.float32)
        conv4_w[0, :] = conv4_w_t.copy()
        del conv4_w_t
        conv4_w[1, :] = tr_wts[10]

        conv4_b_t = tr_wts[9]
        conv4_b = np.empty((2, conv4_b_t.shape[0]), np.float32)
        conv4_b[0, :] = conv4_b_t.copy()
        del conv4_b_t
        conv4_b[1, :] = tr_wts[11]

        conv_4 = merge([
                           Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1),
                                         weights=[conv4_w[i, :], conv4_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

        conv_5 = ZeroPadding2D((1, 1))(conv_4)

        conv5_w_t = tr_wts[12]
        conv5_w = np.empty((2, conv5_w_t.shape[0], conv5_w_t.shape[1], conv5_w_t.shape[2], conv5_w_t.shape[3]),
                           np.float32)
        conv5_w[0, :] = conv5_w_t.copy()
        del conv5_w_t
        conv5_w[1, :] = tr_wts[14]

        conv5_b_t = tr_wts[13]
        conv5_b = np.empty((2, conv5_b_t.shape[0]), np.float32)
        conv5_b[0, :] = conv5_b_t.copy()
        del conv5_b_t
        conv5_b[1, :] = tr_wts[15]

        conv_5 = merge([Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1),
                                         weights=[conv5_w[i, :], conv5_b[i, :]], trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

        dense_1_w = tr_wts[16]
        dense_1_b = tr_wts[17]

        dense_2_w = tr_wts[18]
        dense_2_b = tr_wts[19]

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=trainable)(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=trainable)(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20], tr_wts[21]], trainable=trainable,
                        init='he_normal')(dense_3)
        prediction = Activation("softmax", name="softmax")(dense_3)

    elif layer_id == 3:
        inputs = Input(shape=(384, 13, 13))
        conv_5 = ZeroPadding2D((1, 1))(inputs)

        conv5_w_t = tr_wts[12]
        conv5_w = np.empty((2, conv5_w_t.shape[0], conv5_w_t.shape[1], conv5_w_t.shape[2], conv5_w_t.shape[3]),
                           np.float32)
        conv5_w[0, :] = conv5_w_t.copy()
        del conv5_w_t
        conv5_w[1, :] = tr_wts[14]

        conv5_b_t = tr_wts[13]
        conv5_b = np.empty((2, conv5_b_t.shape[0]), np.float32)
        conv5_b[0, :] = conv5_b_t.copy()
        del conv5_b_t
        conv5_b[1, :] = tr_wts[15]

        conv_5 = merge([ Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1),
                                         weights=[conv5_w[i, :], conv5_b[i, :]],trainable=trainable)(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

        dense_1_w = tr_wts[16]
        dense_1_b = tr_wts[17]

        dense_2_w = tr_wts[18]
        dense_2_b = tr_wts[19]

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=trainable)(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=trainable)(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20], tr_wts[21]], trainable=trainable,
                        init='he_normal')(dense_3)
        prediction = Activation("softmax", name="softmax")(dense_3)

    elif layer_id == 4:
        inputs = Input(shape=(256, 13, 13))
        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(inputs)

        dense_1_w = tr_wts[16]
        dense_1_b = tr_wts[17]

        dense_2_w = tr_wts[18]
        dense_2_b = tr_wts[19]

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=trainable)(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=trainable)(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20], tr_wts[21]], trainable=trainable,
                        init='he_normal')(dense_3)
        prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return model, layer_dict

# critically wide correction unit block
def get_correct_unit_CW(inp, num_correct, kern_sz, num_units, stride=1):
    set_trainable = True
    w_reg = 0.00001
    layer1_c = Convolution2D(num_correct, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                             trainable=set_trainable)(inp)
    layer1_bn = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer1_c)
    layer1_act = Activation('relu')(layer1_bn)
    layer2_pd = ZeroPadding2D(((kern_sz - 1) / 2, (kern_sz - 1) / 2))(layer1_act)
    layer2_c = Convolution2D(num_correct, kern_sz, kern_sz, activation='linear', init='he_normal',
                             W_regularizer=l1(w_reg), trainable=set_trainable, subsample=(stride, stride))(layer2_pd)
    layer2_bn = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer2_c)
    layer2_act = Activation('relu')(layer2_bn)

    for unit_id in range(num_units - 1):
        layer2_act = ZeroPadding2D(((kern_sz - 1) / 2, (kern_sz - 1) / 2))(layer2_act)
        layer2_act = Convolution2D(num_correct, kern_sz, kern_sz, activation='linear', init='he_normal',
                                   W_regularizer=l1(w_reg), trainable=set_trainable, subsample=(stride, stride))(layer2_act)
        layer2_act = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer2_act)
        layer2_act = Activation('relu')(layer2_act)
    unit_out = Convolution2D(num_correct, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                             trainable=set_trainable)(layer2_act)
    return unit_out

# bottleneck and lite correction unit block
def get_correct_unit_bottleneck(inp, inp_d, out_d, dil_f, num_units, stride=1):
    set_trainable = True
    w_reg = 0.00001
    layer1_c = Convolution2D(inp_d, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                             trainable=set_trainable)(inp)
    layer1_bn = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer1_c)
    layer1_act = Activation('relu')(layer1_bn)
    layer2_pd = ZeroPadding2D((dil_f[0], dil_f[0]))(layer1_act)
    layer2_c = AtrousConvolution2D(inp_d, 3, 3, activation='linear', init='he_normal', atrous_rate=(dil_f[0], dil_f[0]),
                                   W_regularizer=l1(w_reg), trainable=True)(layer2_pd)

    layer2_bn = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer2_c)
    layer2_act = Activation('relu')(layer2_bn)

    for unit_id in range(num_units - 1):
        layer2_act = ZeroPadding2D((dil_f[unit_id + 1], dil_f[unit_id + 1]))(layer2_act)
        layer2_act = AtrousConvolution2D(inp_d, 3, 3, activation='linear', init='he_normal',
                                         atrous_rate=(dil_f[unit_id + 1], dil_f[unit_id + 1]), W_regularizer=l1(w_reg),
                                         trainable=True)(layer2_act)

        layer2_act = BatchNormalization(mode=0, axis=1, trainable=set_trainable)(layer2_act)
        layer2_act = Activation('relu')(layer2_act)
    unit_out = Convolution2D(out_d, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                             trainable=set_trainable)(layer2_act)

    return unit_out

# deepcorr model
def alexnet_correct(weights_path=None, out_dim=1000, dist_type='blur', corr_arch = 'CW', num_ly_corr = 5):
    filter_sz = [96, 256, 384, 384, 256]

    tr_wts = np.load(weights_path)
    inp_img = Input((3, IN_dat.img_crop, IN_dat.img_crop), name='input_img')
    set_trainable = False


    correction_perc = 0.75

    def split_tensor_lower(x):

        inp_shape = K.int_shape(x)
        chn = corrected_chn

        return x[:, chn:, :, :]

    def split_tensor_upper(x):
        # corr_per = 0.25
        inp_shape = K.int_shape(x)
        chn = corrected_chn
        return x[:, :chn, :, :]

    def split_tensor_uppershape(input_shape):
        inp_shape = input_shape
        chn = corrected_chn
        shape = (chn,) + input_shape[2:]
        shape = (input_shape[0],) + shape
        return shape

    def split_tensor_lowershape(input_shape):
        inp_shape = input_shape
        chn = inp_shape[1] - corrected_chn
        shape = (chn,) + input_shape[2:]
        shape = (input_shape[0],) + shape
        return shape
        # else:
        #     def split_tensor_lower(x):
        #
        #         inp_shape = K.int_shape(x)
        #         chn = int(inp_shape[1] * correction_perc)
        #
        #         return x[:, chn:, :, :]
        #
        #     def split_tensor_upper(x):
        #         # corr_per = 0.25
        #         inp_shape = K.int_shape(x)
        #         chn = int(inp_shape[1] * correction_perc)
        #         return x[:, :chn, :, :]
        #
        #     def split_tensor_uppershape(input_shape):
        #         inp_shape = input_shape
        #         chn = int(inp_shape[1] * correction_perc)
        #         shape = (chn,) + input_shape[2:]
        #         shape = (input_shape[0],) + shape
        #         return shape
        #
        #     def split_tensor_lowershape(input_shape):
        #         inp_shape = input_shape
        #         chn = inp_shape[1] - int(inp_shape[1] * correction_perc)
        #         shape = (chn,) + input_shape[2:]
        #         shape = (input_shape[0],) + shape
        #         return shape


    wts_l1 = tr_wts[0].copy()
    b_l1 = tr_wts[1].copy()


    ranked_fltr = h5py.File('imagenet_alexnet_ranked_filters.h5','r')

    if dist_type == 'blur':
        corr_id = ranked_fltr['alexnet_blur/layer_'+str(1)][:]

    elif dist_type == 'awgn':
        corr_id = ranked_fltr['alexnet_awgn/layer_'+str(1)][:]



    corrected_chn = int(correction_perc * filter_sz[0])
    print corrected_chn
    # print corr_id
    wts_l1 = wts_l1[corr_id, :, :, :]
    b_l1 = b_l1[corr_id]

    wts_remap_l1 = np.zeros((filter_sz[0], filter_sz[0], 1, 1), np.float32)
    b_remap_l1 = np.zeros(filter_sz[0], np.float32)

    for filt_id in range(filter_sz[0]):
        wts_remap_l1[corr_id[filt_id], filt_id, :, :] = 1.0

    # inp_img1 = ZeroPadding2D((2,2))(inp_img)
    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='linear', weights=[wts_l1, b_l1],
                           name='conv_1', trainable=set_trainable)(inp_img)

    conv1_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv_1)

    conv1_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv_1)

    if corr_arch=='CW':
        conv1_correct = get_correct_unit_CW(conv1_upper, corrected_chn, 5, 2)
    elif corr_arch=='fixed':
        conv1_correct = get_correct_unit_bottleneck(conv1_upper, 128, corrected_chn, [1, 2], 2)
    else:
        conv1_correct = get_correct_unit_bottleneck(conv1_upper, int(0.5 * corrected_chn), corrected_chn, [1, 1, 2], 3)

    conv1_sum_merge1 = merge([conv1_correct, conv1_upper], mode='sum')

    conv1_1_merged = merge([conv1_sum_merge1, conv1_lower], mode='concat', concat_axis=1)

    conv_1_remap = Convolution2D(filter_sz[0], 1, 1, activation='linear', name='conv_1_corr', trainable=False,
                                 weights=[wts_remap_l1, b_remap_l1])(conv1_1_merged)

    conv_1_out = Activation('relu', name='conv_1_relu')(conv_1_remap)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1_out)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv2_w_t = tr_wts[2]
    conv2_w = np.empty((2, conv2_w_t.shape[0], conv2_w_t.shape[1], conv2_w_t.shape[2], conv2_w_t.shape[3]), np.float32)
    conv2_w[0, :] = conv2_w_t.copy()
    del conv2_w_t
    conv2_w[1, :] = tr_wts[4]

    conv2_b_t = tr_wts[3]
    conv2_b = np.empty((2, conv2_b_t.shape[0]), np.float32)
    conv2_b[0, :] = conv2_b_t.copy()
    del conv2_b_t
    conv2_b[1, :] = tr_wts[5]

    conv_2 = merge([Convolution2D(128, 5, 5, activation="linear", name='conv_2_' + str(i + 1), weights=[conv2_w[i, :],
                    conv2_b[i, :]], trainable=set_trainable)(splittensor(ratio_split=2, id_split=i)(conv_2))
                    for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    if num_ly_corr > 1:

        if dist_type == 'blur':
            corr_id = ranked_fltr['alexnet_blur/layer_' + str(2)][:]

        elif dist_type == 'awgn':
            corr_id = ranked_fltr['alexnet_awgn/layer_' + str(2)][:]


        corrected_chn = int(correction_perc * filter_sz[1])
        print corrected_chn
        # print corr_id

        wts_remap_l2_inp = np.zeros((filter_sz[1], filter_sz[1], 1, 1), np.float32)
        b_remap_l2_inp = np.zeros(filter_sz[1], np.float32)

        wts_remap_l2 = np.zeros((filter_sz[1], filter_sz[1], 1, 1), np.float32)
        b_remap_l2 = np.zeros(filter_sz[1], np.float32)

        for filt_id in range(filter_sz[1]):
            wts_remap_l2_inp[filt_id, corr_id[filt_id], :, :] = 1.0
            wts_remap_l2[corr_id[filt_id], filt_id, :, :] = 1.0

        conv_2_inp_remap = Convolution2D(filter_sz[1], 1, 1, activation='linear', trainable=False,
                                         weights=[wts_remap_l2_inp, b_remap_l2_inp])(conv_2)

        conv2_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv_2_inp_remap)
        conv2_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv_2_inp_remap)

        if corr_arch=='CW':
            conv2_correct = get_correct_unit_CW(conv2_upper, corrected_chn, 3, 2)
        elif corr_arch=='fixed':
            conv2_correct = get_correct_unit_bottleneck(conv2_upper, 128, corrected_chn, [1, 2], 2)
        else:
            conv2_correct = get_correct_unit_bottleneck(conv2_upper, int(0.5 * corrected_chn), corrected_chn, [1, 1, 1], 3)

        conv2_sum_merge1 = merge([conv2_correct, conv2_upper], mode='sum')

        conv2_1_merged = merge([conv2_sum_merge1, conv2_lower], mode='concat', concat_axis=1)

        conv_2_remap = Convolution2D(filter_sz[1], 1, 1, activation='linear', name='conv_2_corr', trainable=False,
                                     weights=[wts_remap_l2, b_remap_l2])(conv2_1_merged)

        conv_2_out = Activation('relu', name='conv_2_relu')(conv_2_remap)
    else:

        conv_2_out = Activation('relu', name='conv_2_relu')(conv_2)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2_out)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)

    wts_l3 = tr_wts[6].copy()
    b_l3 = tr_wts[7].copy()
    #
    if num_ly_corr > 2:
        correction_perc = 0.5

        if dist_type == 'blur':
            corr_id = ranked_fltr['alexnet_blur/layer_' + str(3)][:]

        elif dist_type == 'awgn':
            corr_id = ranked_fltr['alexnet_awgn/layer_' + str(3)][:]

        corrected_chn = int(correction_perc * filter_sz[2])

        print corrected_chn
        # print corr_id
        wts_l3 = wts_l3[corr_id, :, :, :]
        b_l3 = b_l3[corr_id]

        wts_remap_l3 = np.zeros((filter_sz[2], filter_sz[2], 1, 1), np.float32)
        b_remap_l3 = np.zeros(filter_sz[2], np.float32)

        for filt_id in range(filter_sz[2]):
            wts_remap_l3[corr_id[filt_id], filt_id, :, :] = 1.0

        conv_3 = Convolution2D(384, 3, 3, weights=[wts_l3, b_l3], activation='linear', name='conv_3',trainable=set_trainable)(conv_3)

        #
        conv3_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv_3)
        conv3_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv_3)

        if corr_arch == 'CW':
            conv3_correct = get_correct_unit_CW(conv3_upper, corrected_chn, 3, 2)
        elif corr_arch=='fixed':
            conv3_correct = get_correct_unit_bottleneck(conv3_upper, 128, corrected_chn, [1, 1], 2)
        else:
            conv3_correct = get_correct_unit_bottleneck(conv3_upper, int(0.5 * corrected_chn), corrected_chn, [1, 1, 1], 3)



        conv3_sum_merge1 = merge([conv3_correct, conv3_upper], mode='sum')

        conv3_1_merged = merge([conv3_sum_merge1, conv3_lower], mode='concat', concat_axis=1)

        conv_3_remap = Convolution2D(filter_sz[2], 1, 1, activation='linear', name='conv_3_corr', trainable=False,
                                     weights=[wts_remap_l3, b_remap_l3])(conv3_1_merged)

        conv_3_out = Activation('relu', name='conv_3_relu')(conv_3_remap)

    else:
        conv_3 = Convolution2D(384, 3, 3, weights=[wts_l3, b_l3], activation='linear', name='conv_3', trainable=set_trainable)(conv_3)

        conv_3_out = Activation('relu', name='conv_3_relu')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3_out)

    conv4_w_t = tr_wts[8].copy()
    conv4_w = np.empty((2, conv4_w_t.shape[0], conv4_w_t.shape[1], conv4_w_t.shape[2], conv4_w_t.shape[3]), np.float32)
    conv4_w[0, :] = conv4_w_t.copy()
    del conv4_w_t
    conv4_w[1, :] = tr_wts[10].copy()

    conv4_b_t = tr_wts[9].copy()
    conv4_b = np.empty((2, conv4_b_t.shape[0]), np.float32)
    conv4_b[0, :] = conv4_b_t.copy()
    del conv4_b_t
    conv4_b[1, :] = tr_wts[11].copy()

    conv_4 = merge([Convolution2D(192, 3, 3, activation="linear", name='conv_4_' + str(i + 1),
                    weights=[conv4_w[i, :], conv4_b[i, :]],trainable=set_trainable)(splittensor(ratio_split=2, id_split=i)(conv_4))
                    for i in range(2)], mode='concat', concat_axis=1, name="conv_4")
    #
    if num_ly_corr > 3:
        correction_perc = 0.5

        if dist_type == 'blur':
            corr_id = ranked_fltr['alexnet_blur/layer_' + str(4)][:]

        elif dist_type == 'awgn':
            corr_id = ranked_fltr['alexnet_awgn/layer_' + str(4)][:]

        corrected_chn = int(correction_perc * filter_sz[3])
            # corrected_chn = corr_chn
        print corrected_chn
        # print corr_id
        wts_remap_l4_inp = np.zeros((filter_sz[3], filter_sz[3], 1, 1), np.float32)
        b_remap_l4_inp = np.zeros(filter_sz[3], np.float32)

        wts_remap_l4 = np.zeros((filter_sz[3], filter_sz[3], 1, 1), np.float32)
        b_remap_l4 = np.zeros(filter_sz[3], np.float32)

        for filt_id in range(filter_sz[3]):
            wts_remap_l4_inp[filt_id, corr_id[filt_id], :, :] = 1.0
            wts_remap_l4[corr_id[filt_id], filt_id, :, :] = 1.0

        conv_4_inp_remap = Convolution2D(filter_sz[3], 1, 1, activation='linear', trainable=False,
                                         weights=[wts_remap_l4_inp, b_remap_l4_inp])(conv_4)

        conv4_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv_4_inp_remap)
        conv4_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv_4_inp_remap)

        if corr_arch=='CW':
            conv4_correct = get_correct_unit_CW(conv4_upper, corrected_chn, 3, 2)
        elif corr_arch=='fixed':
            conv4_correct = get_correct_unit_bottleneck(conv4_upper, 128, corrected_chn, [1, 1], 2)
        else:
            conv4_correct = get_correct_unit_bottleneck(conv4_upper, int(0.5 * corrected_chn), corrected_chn, [1, 1, 1], 3)

        conv4_sum_merge1 = merge([conv4_correct, conv4_upper], mode='sum')

        conv4_1_merged = merge([conv4_sum_merge1, conv4_lower], mode='concat', concat_axis=1)

        conv_4_remap = Convolution2D(filter_sz[3], 1, 1, activation='linear', name='conv_4_corr', trainable=False,
                                     weights=[wts_remap_l4, b_remap_l4])(conv4_1_merged)

        conv_4_out = Activation('relu', name='conv_4_relu')(conv_4_remap)
    else:

        conv_4_out = Activation('relu', name='conv_4_relu')(conv_4)

    conv_5 = ZeroPadding2D((1, 1))(conv_4_out)

    conv5_w_t = tr_wts[12].copy()
    conv5_w = np.empty((2, conv5_w_t.shape[0], conv5_w_t.shape[1], conv5_w_t.shape[2], conv5_w_t.shape[3]), np.float32)
    conv5_w[0, :] = conv5_w_t.copy()
    del conv5_w_t
    conv5_w[1, :] = tr_wts[14].copy()

    conv5_b_t = tr_wts[13].copy()
    conv5_b = np.empty((2, conv5_b_t.shape[0]), np.float32)
    conv5_b[0, :] = conv5_b_t.copy()
    del conv5_b_t
    conv5_b[1, :] = tr_wts[15].copy()

    conv_5 = merge([Convolution2D(128, 3, 3, activation="linear", name='conv_5_' + str(i + 1),weights=[conv5_w[i, :], conv5_b[i, :]],trainable=set_trainable)
                        (splittensor(ratio_split=2, id_split=i)(conv_5))
                    for i in range(2)], mode='concat', concat_axis=1, name="conv_5")
    #
    #
    #
    if num_ly_corr > 4:
        correction_perc = 0.5

        if dist_type == 'blur':
            corr_id = ranked_fltr['alexnet_blur/layer_' + str(5)][:]

        elif dist_type == 'awgn':
            corr_id = ranked_fltr['alexnet_awgn/layer_' + str(5)][:]

        corrected_chn = int(correction_perc * filter_sz[4])
        print corrected_chn
        # print corr_id

        wts_remap_l5_inp = np.zeros((filter_sz[4], filter_sz[4], 1, 1), np.float32)
        b_remap_l5_inp = np.zeros(filter_sz[4], np.float32)

        wts_remap_l5 = np.zeros((filter_sz[4], filter_sz[4], 1, 1), np.float32)
        b_remap_l5 = np.zeros(filter_sz[4], np.float32)

        for filt_id in range(filter_sz[4]):
            wts_remap_l5_inp[filt_id, corr_id[filt_id], :, :] = 1.0
            wts_remap_l5[corr_id[filt_id], filt_id, :, :] = 1.0

        conv_5_inp_remap = Convolution2D(filter_sz[4], 1, 1, activation='linear', trainable=False,
                                         weights=[wts_remap_l5_inp, b_remap_l5_inp])(conv_5)

        conv5_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv_5_inp_remap)
        conv5_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv_5_inp_remap)

        if corr_arch=='CW':
            conv5_correct = get_correct_unit_CW(conv5_upper, corrected_chn, 3, 2)
        elif corr_arch=='fixed':
            conv5_correct = get_correct_unit_bottleneck(conv5_upper, 128, corrected_chn, [1, 1], 2)
        else:
            conv5_correct = get_correct_unit_bottleneck(conv5_upper, int(0.5 * corrected_chn), corrected_chn, [1, 1, 1], 3)

        conv5_sum_merge1 = merge([conv5_correct, conv5_upper], mode='sum')

        conv5_1_merged = merge([conv5_sum_merge1, conv5_lower], mode='concat', concat_axis=1)

        conv_5_remap = Convolution2D(filter_sz[4], 1, 1, activation='linear', name='conv_5_corr', trainable=False,
                                     weights=[wts_remap_l5, b_remap_l5])(conv5_1_merged)

        conv_5_out = Activation('relu', name='conv_5_relu')(conv_5_remap)
    else:

        conv_5_out = Activation('relu', name='conv_5_relu')(conv_5)

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5_out)

    dense_1_w = tr_wts[16]
    dense_1_b = tr_wts[17]

    dense_2_w = tr_wts[18]
    dense_2_b = tr_wts[19]

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1', weights=[dense_1_w, dense_1_b], trainable=False)(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2', weights=[dense_2_w, dense_2_b], trainable=False)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    dense_3 = Dense(out_dim, name='dense_3', weights=[tr_wts[20][:, :out_dim], tr_wts[21][:out_dim]], trainable=False,
                    init='he_normal')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    print'\n'
    print tr_wts[20].shape
    print '\n'
    print tr_wts[21].shape
    model = Model(input=inp_img, output=prediction)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    return model, layer_dict


