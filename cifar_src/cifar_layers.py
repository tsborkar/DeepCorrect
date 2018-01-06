import numpy as np
import h5py
from keras.optimizers import SGD
from keras import backend as K





img_width, img_height = 32, 32
nchn = 3
layer_names = ['conv1_1','conv1_2','max_pool1','conv2_1','conv2_2','max_pool2','conv3_1','conv3_2','conv3_3','output']

layer_out_names = ['conv1_1_relu','conv1_2_relu','max_pool1_relu','conv2_1_relu','conv2_2_relu','max_pool2_relu','conv3_1_relu','conv3_2_relu','conv3_3_relu']


import cifar_layer_arch as CIFAR_arch


def get_Layer_output():

    model_100, layer_dict_100 = CIFAR_arch.cifar10_100('../cifar_models/CIFAR_100_fine_best_model.npy',img_width,img_height,100)

    model_100.compile(optimizer=SGD(),loss='categorical_crossentropy')


    layer_size100 = np.empty((len(layer_names)-1,3),int)
    layer_size100[0,:] = [96,32,32]
    layer_size100[1,:] = [96,32,32]
    layer_size100[2,:] = [96,16,16]
    layer_size100[3,:] = [192,16,16]
    layer_size100[4,:] = [192,16,16]
    layer_size100[5,:] = [192,8,8]
    layer_size100[6,:] = [192,6,6]
    layer_size100[7,:] = [192,6,6]
    layer_size100[8,:] = [100,6,6]

    print "\n compiled model successfully"

    np.set_printoptions(precision=6,suppress=True)
    num_dist = 6

    data_file = h5py.File('CIFAR_10_100_val.h5','r')

    batch_size  = 1000

    outfile = h5py.File('cifar_layer_outputs.h5','w')

    img_data100 = data_file['img_data100'][:]
    img_labl100 = data_file['img_label100'][:]
    data_file.close()

    for layer_id in range(len(layer_names)-1):

        print '\n layer name : '+str(layer_names[layer_id])
        get_layer_output = K.function([model_100.input,K.learning_phase()],layer_dict_100[layer_out_names[layer_id]].output)
        print'\n Testing ...............'

        filter_op = np.empty((len(img_labl100),layer_size100[layer_id,0],layer_size100[layer_id,1],layer_size100[layer_id,2]),np.float32)
        for img_id in range(0,len(img_labl100)/batch_size):
            print "\n Batch num : " +str(img_id)
            filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([img_data100[img_id*batch_size:(img_id+1)*batch_size,:,:,:],0])
        print filter_op.shape
        outfile.create_dataset('CIFAR_100/'+str(layer_names[layer_id]),data= filter_op)
        print '\n writing layer output data to file '

    del img_data100, img_labl100


    outfile.close()


    for iter_id in range(num_dist):

        outfile_dist = h5py.File('cifar_layer_outputs_dist_'+str(iter_id)+'.h5','w')
        data_file1 = h5py.File('CIFAR_10_100_val_dist_'+str(iter_id)+'.h5','r')


        img_data_dist =data_file1['img_data100'][:]


        for layer_id in range(len(layer_names)-1):


            print '\n layer name : '+str(layer_names[layer_id])
            get_layer_output = K.function([model_100.input,K.learning_phase()],layer_dict_100[layer_out_names[layer_id]].output)

            print'\n Testing ...............'

            filter_op = np.empty((img_data_dist.shape[0],layer_size100[layer_id,0],layer_size100[layer_id,1],layer_size100[layer_id,2]),np.float32)
            for img_id in range(0,img_data_dist.shape[0]/batch_size):
                print "\n Batch num : " + str(img_id)
                filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([img_data_dist[img_id*batch_size:(img_id+1)*batch_size,:,:,:],0])

            outfile_dist.create_dataset('CIFAR_100_dist/'+str(layer_names[layer_id]),data= filter_op)
            print '\n wrinting layer output data to file '
        #
        del img_data_dist

        data_file1.close()
        outfile_dist.close()




        outfile_blur = h5py.File('cifar_layer_outputs_blur_'+str(iter_id)+'.h5','w')
        data_file2 = h5py.File('CIFAR_10_100_val_blur_'+str(iter_id)+'.h5','r')

        img_data_blur = data_file2['img_data100'][:]

        data_file2.close()



        for layer_id in range(len(layer_names)-1):

            print '\n layer name : '+str(layer_names[layer_id])
            get_layer_output = K.function([model_100.input,K.learning_phase()],layer_dict_100[layer_out_names[layer_id]].output)


            print'\n Testing ...............'

            filter_op = np.empty((img_data_blur.shape[0],layer_size100[layer_id,0],layer_size100[layer_id,1],layer_size100[layer_id,2]),np.float32)
            for img_id in range(0,img_data_blur.shape[0]/batch_size):
                print "\n Batch num : " + str(img_id)
                filter_op[img_id*batch_size:(img_id+1)*batch_size,:,:,:] = get_layer_output([img_data_blur[img_id*batch_size:(img_id+1)*batch_size,:,:,:],0])

            outfile_blur.create_dataset('CIFAR_100_blur/'+str(layer_names[layer_id]),data= filter_op)
            print '\n wrinting layer output data to file '
        #


        del img_data_blur

        outfile_blur.close()
