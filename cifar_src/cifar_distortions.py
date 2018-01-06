import numpy as np
import h5py
np.random.seed(1337)

from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

import cifar_layer_arch as CIFAR_arch







img_width, img_height = 32, 32
nchn = 3





def compute_test_accuracy(model,x_test, y_test):
#    print('Testing on CIFAR-10')

    y_pred = model.predict(x_test,batch_size=128)

    top1_err_drop = 0

    for test_id in range(len(y_pred)):

            if(np.argmax(y_pred[test_id,:])!= np.argmax(y_test[test_id,:])):
                top1_err_drop += 1



    accuracy = 1-float(top1_err_drop)/float(len(y_test))
    print '\n Accuracy is : '+str(accuracy)
    return accuracy


def get_baseline_val_acc():

    model_100, layer_dict_100 = CIFAR_arch.cifar10_100('../cifar_models/CIFAR_100_fine_best_model.npy', img_width, img_height, 100)


    model_100.compile(optimizer=SGD(),loss='categorical_crossentropy')



    print "\n compiled model successfully"

    np.set_printoptions(precision=6,suppress=True)
    blur_std = [1,2,3,4,5,6]
    awgn_std = [5,10,20,30,40,50]
    num_dist = 6

    pred_acc100 = np.empty((num_dist,4),np.float32)


    for dist_iter in range(num_dist):

        data_file = h5py.File('CIFAR_10_100_val.h5','r')
        img_data_test = data_file['img_data100'][:]
        img_labl_test = data_file['img_label100'][:]
        img_labl_test = to_categorical(img_labl_test,nb_classes=100)
        data_file.close()

        print '\n Loading CIFAR 100 dataset \n'

        print'\n Testing CIFAR 100 ...............'

        pred_acc100[dist_iter,0] = compute_test_accuracy(model_100,img_data_test,img_labl_test)

        del img_data_test


        data_file1 = h5py.File('CIFAR_10_100_val_dist_'+str(dist_iter)+'.h5','r')
        img_data_dist = data_file1['img_data100'][:]

        data_file1.close()

        pred_acc100[dist_iter,1] = compute_test_accuracy(model_100,img_data_dist,img_labl_test)



        del img_data_dist

        data_file2 = h5py.File('CIFAR_10_100_val_blur_'+str(dist_iter)+'.h5','r')
        img_data_blur =data_file2['img_data100'][:]


        data_file2.close()

        pred_acc100[dist_iter,2] = compute_test_accuracy(model_100,img_data_blur,img_labl_test)



        del img_data_blur, img_labl_test

        print '\n Finished Testing CIFAR 100 '


    print '\n Accuracy for CIFAR 100 '
    print pred_acc100

    print '\n Writing output file'
    outfile = h5py.File('CIFAR_10_100_valacc.h5','w')
    outfile.create_dataset('cifar_100',data=pred_acc100)
    outfile.close()

