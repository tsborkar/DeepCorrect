

import keras
import numpy as np

import h5py



from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, AveragePooling2D, ZeroPadding2D, AtrousConvolution2D
from keras.layers.core import Activation, Flatten, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K

img_width, img_height = 32, 32
nchn = 3
layer_names = ['conv1_1', 'conv1_2', 'max_pool1', 'conv2_1', 'conv2_2', 'max_pool2', 'conv3_1', 'conv3_2',
               'conv3_3', 'output']
layer_out_names = ['conv1_1_relu', 'conv1_2_relu', 'max_pool1_relu', 'conv2_1_relu', 'conv2_2_relu',
                   'max_pool2_relu', 'conv3_1_relu', 'conv3_2_relu', 'conv3_3_relu']
def add_layer(model,out_chn,stride,ly_id,img_w=None,inp_chn=None, do_train = True):


    if len(model.layers)<1:
        model.add(ZeroPadding2D((1,1),input_shape=(inp_chn,img_w,img_w)))
    else:           
        model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(out_chn,3,3,activation='linear',subsample=(stride,stride),name=layer_names[ly_id-1],init='he_normal',W_regularizer=l2(0.0003), trainable=do_train))
    model.add(BatchNormalization(mode=0,axis=1, trainable=do_train))
    model.add(Activation('relu', name=layer_out_names[ly_id-1]))
    return model    
L1_reg = 0.00001

# def cifar10_100(weights_path=None, img_width=32,img_height=32,out_dim=100):
#
#
#     model =  Sequential()
#
#     model = add_layer(model,96,1,1,32,3)
#     model = add_layer(model,96,1,2)
#     model = add_layer(model,96,2,3)
#
#     model = add_layer(model,192,1,4)
#     model = add_layer(model,192,1,5)
#     model = add_layer(model,192,2,6)
#
#
#     model.add(Convolution2D(192,3,3,activation='linear',name='conv3_1',init='he_normal',W_regularizer=l2(0.0003)))
#     model.add(BatchNormalization(mode=0,axis=1))
#     model.add(Activation('relu',name='conv3_1_relu'))
#
#
#     model.add(Convolution2D(192,1,1,activation='linear',name='conv3_2',init='he_normal',W_regularizer=l2(0.0003)))
#     model.add(BatchNormalization(mode=0,axis=1))
#     model.add(Activation('relu',name='conv3_2_relu'))
#
#
#     model.add(Convolution2D(out_dim,1,1,activation='linear',name='conv3_3',init='he_normal',W_regularizer=l2(0.0003)))
#     model.add(BatchNormalization(mode=0,axis=1))
#     model.add(Activation('relu',name='conv3_3_relu'))
#
#     model.add(AveragePooling2D(pool_size=(6,6),strides=(6,6)))
#     model.add(Flatten())
#     model.add(Activation('softmax',name='output'))
#
#     if weights_path:
#         weights = np.load(weights_path)
#         model.set_weights(weights)
# #
#     layer_dict = dict([(layer.name,layer) for layer in model.layers])
#
#     return model, layer_dict

# baseline network arch
def cifar10_100(weights_path=None, img_width=32, img_height=32, out_dim=100,num_ly_corr=6):
    model = Sequential()
    set_trainable = True

    model = add_layer(model, 96, 1,1, 32, 3, do_train = set_trainable)
    if num_ly_corr > 1:
        set_trainable = True
    else:
        set_trainable = False

    model = add_layer(model, 96, 1, 2, do_train = set_trainable)
    if num_ly_corr > 2:
        set_trainable = True
    else:
        set_trainable = False

    model = add_layer(model, 96, 2, 3, do_train = set_trainable)
    if num_ly_corr > 3:
        set_trainable = True
    else:
        set_trainable = False

    model = add_layer(model, 192, 1, 4, do_train = set_trainable)
    if num_ly_corr > 4:
        set_trainable = True
    else:
        set_trainable = False

    model = add_layer(model, 192, 1, 5, do_train = set_trainable)
    if num_ly_corr > 5:
        set_trainable = True
    else:
        set_trainable = False

    model = add_layer(model, 192, 2, 6, do_train = set_trainable)
    if num_ly_corr > 6:
        set_trainable = True
    else:
        set_trainable = False

    model.add(Convolution2D(192, 3, 3, activation='linear', name='conv3_1', init='he_normal', W_regularizer=l2(0.0001),
                            trainable=set_trainable))
    model.add(BatchNormalization(mode=0, axis=1, trainable=set_trainable))
    model.add(Activation('relu'))

    model.add(Convolution2D(192, 1, 1, activation='linear', name='conv3_2', init='he_normal', W_regularizer=l2(0.0001),
                            trainable=set_trainable))
    model.add(BatchNormalization(mode=0, axis=1, trainable=set_trainable))
    model.add(Activation('relu'))

    model.add(
        Convolution2D(out_dim, 1, 1, activation='linear', name='conv3_3', init='he_normal', W_regularizer=l2(0.0001),
                      trainable=set_trainable))
    model.add(BatchNormalization(mode=0, axis=1, trainable=set_trainable))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(6, 6), strides=(6, 6)))
    model.add(Flatten())
    model.add(Activation('softmax', name='output'))

    if weights_path:
        weights = np.load(weights_path)
        model.set_weights(weights)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    return model, layer_dict


def cifar10_100_layer(weights_path, layer_id,out_dim = 100):

    layer_sz = [32,32,32,16,16,16,8,6,6]

    layer_str = [1,1,2,1,1,2,1,1,1]
    layer_outchn = [96,96,96,192,192,192,192,192,10]
    layer_inpchn = [3,96,96,96,192,192,192,192,192]
    
    model  = Sequential()
    for ly_id in range(layer_id,6,1):
        model = add_layer(model,layer_outchn[ly_id],layer_str[ly_id],ly_id+1,layer_sz[ly_id],layer_inpchn[ly_id])
    if layer_id ==6:
        model.add(Convolution2D(192, 3, 3, input_shape=(layer_inpchn[layer_id-1], layer_sz[layer_id],layer_sz[layer_id]), activation='linear', name='conv3_1', init='he_normal', W_regularizer=l2(0.0003)))
    else:
        model.add(Convolution2D(192,3,3,activation='linear',name='conv3_1',init='he_normal',W_regularizer=l2(0.0003)))
    model.add(BatchNormalization(mode=0,axis=1))
    model.add(Activation('relu',name='conv3_1_relu'))


    model.add(Convolution2D(192,1,1,activation='linear',name='conv3_2',init='he_normal',W_regularizer=l2(0.0003)))
    model.add(BatchNormalization(mode=0,axis=1))
    model.add(Activation('relu',name='conv3_2_relu'))


    model.add(Convolution2D(out_dim,1,1,activation='linear',name='conv3_3',init='he_normal',W_regularizer=l2(0.0003)))
    model.add(BatchNormalization(mode=0,axis=1))
    model.add(Activation('relu',name='conv3_3_relu'))

    model.add(AveragePooling2D(pool_size=(6,6),strides=(6,6)))
    model.add(Flatten())
    model.add(Activation('softmax',name='output'))

    if weights_path:
        weights = np.load(weights_path)
        new_wts = []
        for wt_id in range(layer_id*6,len(weights),1):
            new_wts.append(weights[wt_id]) 
        
        model.set_weights(new_wts)
#
    layer_dict = dict([(layer.name,layer) for layer in model.layers])

    return model, layer_dict

# correction unit architecture
def get_correct_unit(inp, num_correct, num_units, dil_f, hidden_dim=64):
    w_reg = L1_reg
    # comment next line for deepcorrect
    #hidden_dim = num_correct
    inp_bn = BatchNormalization(mode=0, axis=1, trainable=True, input_shape=(num_correct, img_height, img_width))(inp)
    layer1_c = Convolution2D(hidden_dim, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                             trainable=True, input_shape=(num_correct, img_height, img_width))(inp_bn)
    layer1_bn = BatchNormalization(mode=0, axis=1, trainable=True)(layer1_c)
    layer1_act = Activation('relu')(layer1_bn)
    layer2_pd = ZeroPadding2D((dil_f[0], dil_f[0]))(layer1_act)
    layer2_c = AtrousConvolution2D(hidden_dim, 3, 3, activation='linear', init='he_normal',
                                   atrous_rate=(dil_f[0], dil_f[0]), W_regularizer=l1(w_reg), trainable=True)(layer2_pd)

    layer2_bn = BatchNormalization(mode=0, axis=1, trainable=True)(layer2_c)
    layer2_act = Activation('relu')(layer2_bn)
    if num_units > 1:
        for stg_id in range(num_units - 1):
            layer2_act = ZeroPadding2D((dil_f[stg_id + 1], dil_f[stg_id + 1]))(layer2_act)
            layer2_act = AtrousConvolution2D(hidden_dim, 3, 3, activation='linear', init='he_normal',atrous_rate=(dil_f[stg_id + 1],
                                        dil_f[stg_id + 1]),W_regularizer=l1(w_reg), trainable=True)(layer2_act)

            layer2_act = BatchNormalization(mode=0, axis=1, trainable=True)(layer2_act)
            layer2_act = Activation('relu')(layer2_act)

    layer4_out = Convolution2D(num_correct, 1, 1, activation='linear', init='he_normal', W_regularizer=l1(w_reg),
                               trainable=True)(layer2_act)
    return layer4_out

# deepcorr model architecture
def get_correct_net(weights_path='../cifar_models/CIFAR_100_fine_best_model.npy', corr_lvl=6, dist_type = 'blur',correction_perc=0.5, out_dim = 100):
    pre_trained_wts = np.load(weights_path)
    correction_perc *= 1.0417
    b_l7 = pre_trained_wts[37]
    g_l7 = pre_trained_wts[38]
    beta_l7 = pre_trained_wts[39]
    mu_l7 = pre_trained_wts[40]
    std_l7 = pre_trained_wts[41]

    # layer 8 wts
    wts_l8 = pre_trained_wts[42]
    b_l8 = pre_trained_wts[43]
    g_l8 = pre_trained_wts[44]
    beta_l8 = pre_trained_wts[45]
    mu_l8 = pre_trained_wts[46]
    std_l8 = pre_trained_wts[47]

    # layer 9 wts
    wts_l9 = pre_trained_wts[48]
    b_l9 = pre_trained_wts[49]
    g_l9 = pre_trained_wts[50]
    beta_l9 = pre_trained_wts[51]
    mu_l9 = pre_trained_wts[52]
    std_l9 = pre_trained_wts[53]

    ranked_fltr = h5py.File('cifar_ranked_filters.h5','r')


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
        # print shape
        return shape

    def split_tensor_lowershape(input_shape):
        inp_shape = input_shape
        chn = inp_shape[1] - corrected_chn
        shape = (chn,) + input_shape[2:]
        shape = (input_shape[0],) + shape
        # print shape
        return shape

    inp_img = Input((3, img_width, img_height), name='input')



    wts_l1 = pre_trained_wts[0]
    b_l1 = pre_trained_wts[1]
    g_l1 = pre_trained_wts[2]
    beta_l1 = pre_trained_wts[3]
    mu_l1 = pre_trained_wts[4]
    std_l1 = pre_trained_wts[5]

    if dist_type == 'blur':
        corr_id =  ranked_fltr['CIFAR_100_blur/layer_'+str(1)][:]
    else:
        corr_id =  ranked_fltr['CIFAR_100_awgn/layer_'+str(1)][:]

    # print corr_id


    corrected_chn = int(correction_perc * 96)
    # print corrected_chn

    g_l1 = g_l1[corr_id]
    beta_l1 = beta_l1[corr_id]
    mu_l1 = mu_l1[corr_id]
    std_l1 = std_l1[corr_id]

    wts_l1 = wts_l1[corr_id, :, :, :]
    b_l1 = b_l1[corr_id]
    w_reg = 0.0001
    inp_img_pad = ZeroPadding2D((1, 1))(inp_img)  # (corrected_inp_act)
    conv1_out = Convolution2D(96, 3, 3, activation='linear', weights=[wts_l1, b_l1], init='he_normal',
                              W_regularizer=l2(w_reg), trainable=False)(inp_img_pad)
    conv1_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv1_out)
    conv1_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                      weights=[g_l1[corrected_chn:], beta_l1[corrected_chn:], mu_l1[corrected_chn:],
                                               std_l1[corrected_chn:]])(conv1_lower)
    conv1_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv1_out)

    conv1_correct = get_correct_unit(conv1_upper, corrected_chn, 2, [1, 1])
    conv1_sum_merge1 = merge([conv1_correct, conv1_upper], mode='sum')

    conv1_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                      weights=[g_l1[:corrected_chn], beta_l1[:corrected_chn], mu_l1[:corrected_chn],
                                               std_l1[:corrected_chn]])(conv1_sum_merge1)


    conv1_conc_merge = merge([conv1_bn_upp, conv1_bn_low], mode='concat', concat_axis=1)

    conv1_act = Activation('relu')(conv1_conc_merge)

    # create layer 2


    wts_l2 = pre_trained_wts[6]
    b_l2 = pre_trained_wts[7]
    g_l2 = pre_trained_wts[8]
    beta_l2 = pre_trained_wts[9]
    mu_l2 = pre_trained_wts[10]
    std_l2 = pre_trained_wts[11]
    wts_l2 = wts_l2[:, corr_id, :, :]

    conv2_pd = ZeroPadding2D((1, 1))(conv1_act)
    wts_l3 = pre_trained_wts[12]

    if corr_lvl > 1:

        if dist_type == 'blur':
            corr_id = ranked_fltr['CIFAR_100_blur/layer_' + str(2)][:]
        else:
            corr_id = ranked_fltr['CIFAR_100_awgn/layer_' + str(2)][:]

        corrected_chn = int(correction_perc * 96)
        print corrected_chn


        wts_l2 = wts_l2[corr_id, :, :, :]
        b_l2 = b_l2[corr_id]
        g_l2 = g_l2[corr_id]
        beta_l2 = beta_l2[corr_id]
        mu_l2 = mu_l2[corr_id]
        std_l2 = std_l2[corr_id]
        wts_l3 = wts_l3[:, corr_id, :, :]

        conv2_out = Convolution2D(96, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l2, b_l2],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv2_pd)
        conv2_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv2_out)
        conv2_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                          weights=[g_l2[corrected_chn:], beta_l2[corrected_chn:], mu_l2[corrected_chn:],
                                                   std_l2[corrected_chn:]])(conv2_lower)

        conv2_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv2_out)

        conv2_correct = get_correct_unit(conv2_upper, corrected_chn, 2, [1, 1])

        conv2_sum_merge1 = merge([conv2_correct, conv2_upper], mode='sum')

        conv2_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                          weights=[g_l2[:corrected_chn], beta_l2[:corrected_chn], mu_l2[:corrected_chn],
                                                   std_l2[:corrected_chn]])(conv2_sum_merge1)

        conv2_bn = merge([conv2_bn_upp, conv2_bn_low], mode='concat', concat_axis=1)

    else:
        conv2_out = Convolution2D(96, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l2, b_l2],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv2_pd)
        conv2_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l2, beta_l2, mu_l2, std_l2])(
            conv2_out)
    conv2_act = Activation('relu')(conv2_bn)

    # create layer 3

    # layer 3 wts

    b_l3 = pre_trained_wts[13]
    g_l3 = pre_trained_wts[14]
    beta_l3 = pre_trained_wts[15]
    mu_l3 = pre_trained_wts[16]
    std_l3 = pre_trained_wts[17]

    conv3_pd = ZeroPadding2D((1, 1))(conv2_act)
    wts_l4 = pre_trained_wts[18]

    if corr_lvl > 2:

        if dist_type == 'blur':
            corr_id = ranked_fltr['CIFAR_100_blur/layer_' + str(3)][:]
        else:
            corr_id = ranked_fltr['CIFAR_100_awgn/layer_' + str(3)][:]

        corrected_chn = int(correction_perc * 96)


        wts_l3 = wts_l3[corr_id, :, :, :]
        b_l3 = b_l3[corr_id]
        g_l3 = g_l3[corr_id]
        beta_l3 = beta_l3[corr_id]
        mu_l3 = mu_l3[corr_id]
        std_l3 = std_l3[corr_id]
        wts_l4 = wts_l4[:, corr_id, :, :]

        conv3_out = Convolution2D(96, 3, 3, activation='linear', subsample=(2, 2), weights=[wts_l3, b_l3],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv3_pd)
        conv3_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv3_out)
        conv3_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                          weights=[g_l3[corrected_chn:], beta_l3[corrected_chn:], mu_l3[corrected_chn:],
                                                   std_l3[corrected_chn:]])(conv3_lower)

        conv3_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv3_out)

        conv3_correct = get_correct_unit(conv3_upper, corrected_chn, 2, [1, 1])
        conv3_sum_merge1 = merge([conv3_correct, conv3_upper], mode='sum')

        conv3_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                          weights=[g_l3[:corrected_chn], beta_l3[:corrected_chn], mu_l3[:corrected_chn],
                                                   std_l3[:corrected_chn]])(conv3_sum_merge1)

        conv3_bn = merge([conv3_bn_upp, conv3_bn_low], mode='concat', concat_axis=1)

    else:
        conv3_out = Convolution2D(96, 3, 3, activation='linear', subsample=(2, 2), weights=[wts_l3, b_l3],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv3_pd)
        conv3_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l3, beta_l3, mu_l3, std_l3])(
            conv3_out)
    conv3_act = Activation('relu')(conv3_bn)

    # create layer 4
    b_l4 = pre_trained_wts[19]
    g_l4 = pre_trained_wts[20]
    beta_l4 = pre_trained_wts[21]
    mu_l4 = pre_trained_wts[22]
    std_l4 = pre_trained_wts[23]
    conv4_pd = ZeroPadding2D((1, 1))(conv3_act)

    wts_l5 = pre_trained_wts[24]
    # corrected_chn = int(correction_perc*192)
    if corr_lvl > 3:

        if dist_type == 'blur':
            corr_id = ranked_fltr['CIFAR_100_blur/layer_' + str(4)][:]
        else:
            corr_id = ranked_fltr['CIFAR_100_awgn/layer_' + str(4)][:]

        corrected_chn = int(correction_perc * 192)

        wts_l4 = wts_l4[corr_id, :, :, :]
        b_l4 = b_l4[corr_id]
        g_l4 = g_l4[corr_id]
        beta_l4 = beta_l4[corr_id]
        mu_l4 = mu_l4[corr_id]
        std_l4 = std_l4[corr_id]
        wts_l5 = wts_l5[:, corr_id, :, :]
        conv4_out = Convolution2D(192, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l4, b_l4],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv4_pd)
        conv4_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv4_out)
        conv4_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                          weights=[g_l4[corrected_chn:], beta_l4[corrected_chn:], mu_l4[corrected_chn:],
                                                   std_l4[corrected_chn:]])(conv4_lower)

        conv4_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv4_out)

        conv4_correct = get_correct_unit(conv4_upper, corrected_chn, 2, [1, 1])
        conv4_sum_merge1 = merge([conv4_correct, conv4_upper], mode='sum')

        conv4_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                          weights=[g_l4[:corrected_chn], beta_l4[:corrected_chn], mu_l4[:corrected_chn],
                                                   std_l4[:corrected_chn]])(conv4_sum_merge1)

        conv4_bn = merge([conv4_bn_upp, conv4_bn_low], mode='concat', concat_axis=1)

    else:
        conv4_out = Convolution2D(192, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l4, b_l4],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv4_pd)
        conv4_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l4, beta_l4, mu_l4, std_l4])(
            conv4_out)
    conv4_act = Activation('relu')(conv4_bn)

    # create layer 5
    b_l5 = pre_trained_wts[25]
    g_l5 = pre_trained_wts[26]
    beta_l5 = pre_trained_wts[27]
    mu_l5 = pre_trained_wts[28]
    std_l5 = pre_trained_wts[29]
    wts_l6 = pre_trained_wts[30]

    conv5_pd = ZeroPadding2D((1, 1))(conv4_act)

    if corr_lvl > 4:

        if dist_type == 'blur':
            corr_id = ranked_fltr['CIFAR_100_blur/layer_' + str(5)][:]
        else:
            corr_id = ranked_fltr['CIFAR_100_awgn/layer_' + str(5)][:]


        corrected_chn = int(correction_perc * 192)


        wts_l5 = wts_l5[corr_id, :, :, :]
        b_l5 = b_l5[corr_id]
        g_l5 = g_l5[corr_id]
        beta_l5 = beta_l5[corr_id]
        mu_l5 = mu_l5[corr_id]
        std_l5 = std_l5[corr_id]
        wts_l6 = wts_l6[:, corr_id, :, :]

        conv5_out = Convolution2D(192, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l5, b_l5],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv5_pd)
        conv5_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv5_out)
        conv5_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                          weights=[g_l5[corrected_chn:], beta_l5[corrected_chn:], mu_l5[corrected_chn:],
                                                   std_l5[corrected_chn:]])(conv5_lower)

        conv5_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv5_out)

        conv5_correct = get_correct_unit(conv5_upper, corrected_chn, 2, [1, 1])
        conv5_sum_merge1 = merge([conv5_correct, conv5_upper], mode='sum')

        conv5_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                          weights=[g_l5[:corrected_chn], beta_l5[:corrected_chn], mu_l5[:corrected_chn],
                                                   std_l5[:corrected_chn]])(conv5_sum_merge1)


        conv5_bn = merge([conv5_bn_upp, conv5_bn_low], mode='concat', concat_axis=1)
    else:
        conv5_out = Convolution2D(192, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l5, b_l5],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv5_pd)
        conv5_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l5, beta_l5, mu_l5, std_l5])(
            conv5_out)
    conv5_act = Activation('relu')(conv5_bn)

    # create layer 6

    b_l6 = pre_trained_wts[31]
    g_l6 = pre_trained_wts[32]
    beta_l6 = pre_trained_wts[33]
    mu_l6 = pre_trained_wts[34]
    std_l6 = pre_trained_wts[35]
    wts_l7 = pre_trained_wts[36]
    conv6_pd = ZeroPadding2D((1, 1))(conv5_act)

    if corr_lvl > 5:

        if dist_type == 'blur':
            corr_id = ranked_fltr['CIFAR_100_blur/layer_' + str(6)][:]
        else:
            corr_id = ranked_fltr['CIFAR_100_awgn/layer_' + str(6)][:]


        corrected_chn = int(correction_perc * 192)

        wts_l6 = wts_l6[corr_id, :, :, :]
        b_l6 = b_l6[corr_id]
        g_l6 = g_l6[corr_id]
        beta_l6 = beta_l6[corr_id]
        mu_l6 = mu_l6[corr_id]
        std_l6 = std_l6[corr_id]
        wts_l7 = wts_l7[:, corr_id, :, :]

        conv6_out = Convolution2D(192, 3, 3, activation='linear', subsample=(2, 2), weights=[wts_l6, b_l6],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv6_pd)
        conv6_lower = Lambda(split_tensor_lower, output_shape=split_tensor_lowershape)(conv6_out)
        conv6_bn_low = BatchNormalization(mode=0, axis=1, trainable=False,
                                          weights=[g_l6[corrected_chn:], beta_l6[corrected_chn:], mu_l6[corrected_chn:],
                                                   std_l6[corrected_chn:]])(conv6_lower)

        conv6_upper = Lambda(split_tensor_upper, output_shape=split_tensor_uppershape)(conv6_out)

        conv6_correct = get_correct_unit(conv6_upper, corrected_chn, 2, [1, 1])
        conv6_sum_merge1 = merge([conv6_correct, conv6_upper], mode='sum')


        conv6_bn_upp = BatchNormalization(mode=0, axis=1, trainable=True,
                                          weights=[g_l6[:corrected_chn], beta_l6[:corrected_chn], mu_l6[:corrected_chn],
                                                   std_l6[:corrected_chn]])(conv6_sum_merge1)

        conv6_bn = merge([conv6_bn_upp, conv6_bn_low], mode='concat', concat_axis=1)

    else:
        conv6_out = Convolution2D(192, 3, 3, activation='linear', subsample=(2, 2), weights=[wts_l6, b_l6],
                                  init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv6_pd)
        conv6_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l6, beta_l6, mu_l6, std_l6])(
            conv6_out)
    conv6_act = Activation('relu')(conv6_bn)

    # create layer 7

    conv7_out = Convolution2D(192, 3, 3, activation='linear', subsample=(1, 1), weights=[wts_l7, b_l7],
                              init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv6_act)
    conv7_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l7, beta_l7, mu_l7, std_l7])(conv7_out)
    conv7_act = Activation('relu')(conv7_bn)

    # create layer 8

    conv8_out = Convolution2D(192, 1, 1, activation='linear', subsample=(1, 1), weights=[wts_l8, b_l8],
                              init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv7_act)
    conv8_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l8, beta_l8, mu_l8, std_l8])(conv8_out)
    conv8_act = Activation('relu')(conv8_bn)

    # create layer 9

    conv9_out = Convolution2D(out_dim, 1, 1, activation='linear', subsample=(1, 1), weights=[wts_l9, b_l9],
                              init='he_normal', W_regularizer=l2(w_reg), trainable=False)(conv8_act)
    conv9_bn = BatchNormalization(mode=0, axis=1, trainable=False, weights=[g_l9, beta_l9, mu_l9, std_l9])(conv9_out)
    conv9_act = Activation('relu')(conv9_bn)

    avg_pool = AveragePooling2D(pool_size=(6, 6), strides=(6, 6))(conv9_act)
    flat_out = Flatten()(avg_pool)
    net_out = Activation('softmax', name='output')(flat_out)
    model = Model(input=inp_img, output=net_out)
    return model

