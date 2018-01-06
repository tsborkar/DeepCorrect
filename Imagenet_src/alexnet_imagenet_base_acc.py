import numpy as np
import h5py
import cv2
from os import listdir

from keras.optimizers import SGD
from keras import backend as K

import Imagenet_datagen as IN_dat
import alexnet_layer_arch as alexnet

def compute_test_accuracy(y_pred, img_labl):
    top1_err = 0
    top5_err = 0
    for val_id in range(y_pred.shape[0]):
        assert(np.isnan(np.sum(y_pred[val_id,:]))==0),"\n Nan value found in prediction labels"


        # print '\n class label '+str(np.argmax(y_pred[val_id,:]))+'\t GT label '+str(img_labl[val_id,0])
        if (np.argmax(y_pred[val_id, :]) != img_labl[val_id]):
            top1_err += 1



    for val_id in range(y_pred.shape[0]):
        y_sort = np.sort(y_pred[val_id, :])[::-1]
        assert(np.isnan(np.sum(y_sort))==0),"\n Nan value found in sorted prediction labels"


        err_flag = 0

        for s_id in range(5):
            if (int(np.nonzero(y_pred[val_id, :] == y_sort[s_id])[0][0]) == int(img_labl[val_id])):
                err_flag = 1
                break
        if err_flag == 0:
            top5_err += 1

    top1_acc = 1.0 -(top1_err)/float(y_pred.shape[0])
    top5_acc = 1.0 -(top5_err)/float(y_pred.shape[0])
    print '\n top 1 acc : '+str(top1_acc)
    print '\n top 5 acc : '+str(top5_acc)
    return top1_acc, top5_acc




def dc_ranksubsetacc():
    # get baseline AlexNet model
    model_101, layer_dict_101 = alexnet.AlexNetDNN(weights_path='../Imagenet_models/alexnet_weights.h5', heatmap=False,
                                                   out_dim=IN_dat.num_classes)
    model_101.compile(optimizer=SGD(), loss='categorical_crossentropy')

    # build function to get AlexNet FC2 layer output
    get_layer_output = K.function([model_101.input, K.learning_phase()], layer_dict_101['dense_2'].output)

    acc_mat  = np.empty((2,IN_dat.num_dist,3), np.float32)

    inp_file = h5py.File('imagenet_dcranksubset_ori.h5','r')
    img_data = inp_file['img_data'][:]
    img_label = inp_file['img_label'][:]

    y_pred = model_101.predict(img_data,batch_size=512)

    acc_mat [0,:,0], acc_mat[1,:,0] = compute_test_accuracy(y_pred,img_label)

    del img_data, img_label
    inp_file.close()

    for dist_id in range(IN_dat.num_dist):
        print '\n Testing awgn level '+str(dist_id)
        inp_file = h5py.File('imagenet_dcranksubset_awgn_' + str(dist_id) + '.h5', 'r')
        img_data = inp_file['img_data'][:]
        img_label = inp_file['img_label'][:]

        y_pred = model_101.predict(img_data, batch_size=512)

        acc_mat[0,dist_id,1], acc_mat[1,dist_id,1] = compute_test_accuracy(y_pred,img_label)

        del img_data, img_label
        inp_file.close()

        print '\n Testing blur level '+str(dist_id)
        inp_file = h5py.File('imagenet_dcranksubset_blur_' + str(dist_id) + '.h5', 'r')
        img_data = inp_file['img_data'][:]
        img_label  = inp_file['img_label'][:]

        y_pred = model_101.predict(img_data,batch_size=512)
        acc_mat[0,dist_id,2], acc_mat[1,dist_id,2] = compute_test_accuracy(y_pred, img_label)

        del img_data, img_label
        inp_file.close()


    print '\n ------------------Top 1 acc---------------------------\n'
    print acc_mat[0,:,:]
    print '\n ------------------Top 5 acc-----------------------------\n '
    print acc_mat[1,:,:]

    out_file = h5py.File('imagenet_alexnet_ref_acc.h5','w')
    out_file.create_dataset('acc_mat', data=acc_mat)
    print '\n finished writing results to file'
    out_file.close()



