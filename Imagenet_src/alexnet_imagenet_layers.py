import numpy as np
import h5py

from keras.optimizers import SGD
from keras import backend as K


import Imagenet_datagen as IN_dat
import alexnet_layer_arch as alexnet

np.set_printoptions(precision=6,suppress=True)



def get_alexnet_IN_layerout():
    num_val_small = 5
    model_101, layer_dict_101 = alexnet.AlexNetDNN(weights_path='../Imagenet_models/alexnet_weights.h5',heatmap=False,out_dim=IN_dat.num_classes)
    model_101.compile(optimizer=SGD(),loss='categorical_crossentropy')
    model_wts = model_101.get_weights()
    np.save('../Imagenet_models/alexnet_imagenet_weights.npy',model_wts)

    batch_size  = num_val_small*IN_dat.num_classes/10
    nchn=3

    outfile = h5py.File('alexnet_imagenet_layer_outputs.h5','w')
    inp_file = h5py.File('imagenet_dcranksubset_ori.h5','r')
    xval = inp_file['img_data'][:]
    for layer_id in range(0,5,1):

        print '\n layer name : '+str(alexnet.layer_names[layer_id])
        get_layer_output = K.function([model_101.input,K.learning_phase()],layer_dict_101[alexnet.layer_names[layer_id]].output)


        print'\n Testing ...............'
        filter_op = np.empty((num_val_small*IN_dat.num_classes,alexnet.layer_size101[layer_id,0],
                              alexnet.layer_size101[layer_id,1],alexnet.layer_size101[layer_id,2]),np.float32)

        for img_id in range(0,(num_val_small*IN_dat.num_classes)/batch_size):
            print "\n Batch num : "+str(img_id)
            temp_batch = xval[img_id*batch_size:(img_id+1)*batch_size,:,:,:].copy()
            filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([temp_batch,0])
            del temp_batch
        print filter_op.shape
        outfile.create_dataset(str(alexnet.layer_names[layer_id]),data= filter_op)
        print '\n writing layer output data to file '

    del xval
    inp_file.close()
    outfile.close()



    for iter_id in range(IN_dat.num_dist):

        outfile_dist = h5py.File('alexnet_imagenet_layer_outputs_awgn_'+str(iter_id)+'.h5','w')
        inp_file = h5py.File('imagenet_dcranksubset_awgn_' + str(iter_id) + '.h5','r')
        xval = inp_file['img_data'][:]
        for layer_id in range(0,5,1):


            print '\n layer name : '+str(alexnet.layer_names[layer_id])
            get_layer_output = K.function([model_101.input,K.learning_phase()],layer_dict_101[alexnet.layer_names[layer_id]].output)


            print'\n Testing ...............'

            filter_op = np.empty((IN_dat.num_classes*num_val_small,alexnet.layer_size101[layer_id,0],
                                  alexnet.layer_size101[layer_id,1],alexnet.layer_size101[layer_id,2]),np.float32)
            for img_id in range(0,(IN_dat.num_classes*num_val_small)/batch_size):
                print "\n Batch num : "+str(img_id)
                temp_batch = xval[img_id*batch_size:(img_id+1)*batch_size,:,:,:].copy()
                filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([temp_batch,0])

            outfile_dist.create_dataset(str(alexnet.layer_names[layer_id]),data= filter_op)
            print '\n wrinting layer output data to file '
        #
        del xval
        outfile_dist.close()
        inp_file.close()



        outfile_blur = h5py.File('alexnet_imagenet_layer_outputs_blur_'+str(iter_id)+'.h5','w')
        inp_file = h5py.File('imagenet_dcranksubset_blur_'+str(iter_id)+'.h5','r')
        xval = inp_file['img_data'][:]

        for layer_id in range(0,5,1):
            #layer_name = 'conv1_1'

            print '\n layer name : '+str(alexnet.layer_names[layer_id])
            get_layer_output = K.function([model_101.input,K.learning_phase()],layer_dict_101[alexnet.layer_names[layer_id]].output)


            print'\n Testing ...............'

            filter_op = np.empty((IN_dat.num_classes*num_val_small,alexnet.layer_size101[layer_id,0],alexnet.layer_size101[layer_id,1],alexnet.layer_size101[layer_id,2]),np.float32)
            for img_id in range(0,(num_val_small*IN_dat.num_classes)/batch_size):
                print "\n Batch num : " + str(img_id)

                temp_batch = xval[img_id*batch_size:(img_id+1)*batch_size,:,:,:].copy()
                filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([temp_batch,0])

            outfile_blur.create_dataset(str(alexnet.layer_names[layer_id]),data= filter_op)
            print '\n wrinting layer output data to file '
        #


        outfile_blur.close()
        del xval
        inp_file.close()


        print '\n finished writing data'
        #    del img_data_dist


