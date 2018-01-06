
import numpy as np
import h5py
import cv2
from keras.datasets import cifar100


def gen_cifar_val_data():
    # load cifar 100 fine  and compute mean and var
    (X_train100,Y_train100),(X_test100,Y_test100) = cifar100.load_data(label_mode='fine')
    # perm_ids = np.random.permutation(len(X_train100))

    # X_train100 = X_train100[perm_ids,:]
    X_train100  = X_train100.astype(float,copy=False)


    mean_r_100 = np.mean(X_train100[:,0,:,:])
    mean_g_100 = np.mean(X_train100[:,1,:,:])
    mean_b_100 = np.mean(X_train100[:,2,:,:])

    std_r_100 = np.std(X_train100[:,0,:,:])
    std_g_100 = np.std(X_train100[:,1,:,:])
    std_b_100 = np.std(X_train100[:,2,:,:])

    # del X_train100, Y_train100

    nsamp_per_cl = 10
    num_train = nsamp_per_cl*100

    img_width, img_height = 32, 32
    nchn = 3


    blur_std =[0.5,1.,1.5,2.,2.5,3.]
    blur_win = [3,5,7,9,11,13]
    awgn_std =[5,10,15,20,25,30]

    dist_iter = 6
    img_data_ori100 = np.empty((num_train,nchn, img_width,img_height), np.float32)
    img_labl_ori100 = np.empty(num_train,np.float32)
    for iter_id in range(dist_iter):


        img_data_dist100 =  np.empty((num_train,nchn, img_width,img_height), np.float32)
        img_data_blur100 =  np.empty((num_train,nchn, img_width,img_height), np.float32)
        np.random.seed(1337)
        for cl_id in range(100):
            perm_ids = np.random.permutation(500)
            for img_id in range(nsamp_per_cl):

                image_id = cl_id*500+perm_ids[img_id]
                img_temp = np.copy(X_train100[image_id,:,:,:])
                if iter_id==0:
                    img_data_ori100[cl_id * nsamp_per_cl + img_id, :, :, :] = img_temp
                    img_labl_ori100[cl_id * nsamp_per_cl + img_id] = Y_train100[image_id]
                img_temp1 = np.empty((img_width, img_height, nchn), np.float32)

                img_temp1[:, :, 0], img_temp1[:, :, 1], img_temp1[:, :, 2] = img_temp[0, :, :], img_temp[1, :, :], img_temp[2, :, :]
                img_temp1 =  cv2.GaussianBlur(img_temp1, (blur_win[iter_id], blur_win[iter_id]), blur_std[iter_id], None,blur_std[iter_id], cv2.BORDER_CONSTANT)
                img_data_blur100[cl_id*nsamp_per_cl+img_id, 0, :, :], img_data_blur100[cl_id*nsamp_per_cl+img_id, 1, :, :], img_data_blur100[cl_id*nsamp_per_cl+img_id, 2, :,:] \
                    = img_temp1[:, :, 0], img_temp1[:, :,1], img_temp1[:, :,2]
                del img_temp
                img_temp = np.copy(X_train100[image_id,:,:,:])
                img_data_dist100[cl_id*nsamp_per_cl+img_id,:,:,:] = img_temp  + np.random.normal(0,awgn_std[iter_id],(3,img_width,img_height))
                del img_temp



        img_data_dist100[:,0,:,:] -= mean_r_100
        img_data_dist100[:,1,:,:] -= mean_g_100
        img_data_dist100[:,2,:,:] -= mean_b_100

        img_data_dist100[:,0,:,:] /= std_r_100
        img_data_dist100[:,1,:,:] /= std_g_100
        img_data_dist100[:,2,:,:] /= std_b_100

        img_data_blur100[:,0,:,:] -= mean_r_100
        img_data_blur100[:,1,:,:] -= mean_g_100
        img_data_blur100[:,2,:,:] -= mean_b_100

        img_data_blur100[:,0,:,:] /= std_r_100
        img_data_blur100[:,1,:,:] /= std_g_100
        img_data_blur100[:,2,:,:] /= std_b_100




        data_file = h5py.File('CIFAR_10_100_val_'+'dist_'+str(iter_id)+'.h5','w')


        data_file.create_dataset("img_data100",data=img_data_dist100)
        data_file.create_dataset("img_label100",data=img_labl_ori100)


        data_file.close()

        data_file = h5py.File('CIFAR_10_100_val_'+'blur_'+str(iter_id)+'.h5','w')


        data_file.create_dataset("img_data100",data=img_data_blur100)
        data_file.create_dataset("img_label100",data=img_labl_ori100)


        data_file.close()


        print "\n Finished writing image data file"




    data_file = h5py.File('CIFAR_10_100_val.h5','w')



    img_data_ori100[:,0,:] -= mean_r_100
    img_data_ori100[:,1,:] -= mean_g_100
    img_data_ori100[:,2,:] -= mean_b_100
    #
    img_data_ori100[:,0,:] /= std_r_100
    img_data_ori100[:,1,:] /= std_g_100
    img_data_ori100[:,2,:] /= std_b_100
    #



    data_file.create_dataset("img_data100",data=img_data_ori100)
    data_file.create_dataset("img_label100",data=img_labl_ori100)



    data_file.close()


