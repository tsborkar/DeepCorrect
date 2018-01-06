import numpy as np
import h5py
import cv2
from os import listdir

val_folder_loc = '../ILSVRC_data/Class_'
num_classes = 1000
num_val = 10
num_test = 40
num_dist = 6
awgn_std = [10, 20, 40, 60, 80, 100]
blur_std = [1, 2, 3, 4, 5, 6]
img_sz = 256
img_crop = 227
np.random.seed(1337)
rand_idx = np.random.permutation(num_val + num_test)


def imagenet_datagen():



    img_data = np.empty((num_classes*num_val,3,img_crop,img_crop),np.float32)
    img_data_1 = np.empty((num_classes*num_val/2,3,img_crop,img_crop), np.float32)
    img_label = np.empty(num_classes*num_val,np.float32)
    img_label_1 = np.empty(num_classes*num_val/2, np.float32)

    if num_classes==1000:
        out_file1 = h5py.File('imagenet_val_ori.h5','w')
        out_file2 = h5py.File('imagenet_dcranksubset_ori.h5', 'w')
    else:
        out_file1 = h5py.File('imagenet_val_ori_'+str(num_classes)+'.h5','w')
        out_file2 = h5py.File('imagenet_dcranksubset_ori_'+str(num_classes)+'.h5','w')

    for class_id in range(num_classes):

        filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
        print "\n class id"+str(class_id)
        for img_id in range(num_val):

            img_temp = cv2.imread(val_folder_loc + str(class_id) + '/'+filelist[rand_idx[img_id]])
            img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB)
            img_temp = cv2.resize(img_temp, (img_sz,img_sz))
            img_temp = np.transpose(img_temp,(2,0,1))
            img_temp = img_temp[:, (img_sz - img_crop) // 2:(img_sz + img_crop) // 2, (img_sz - img_crop) // 2:(img_sz + img_crop) // 2]
            img_data[num_val*class_id+img_id,:,:,:] = img_temp.copy()
            img_label[num_val*class_id + img_id] = class_id
            if img_id <5:
                img_data_1[num_val * class_id/2 + img_id, :, :, :] = img_temp.copy()
                img_label_1[num_val * class_id/2 + img_id] = class_id


    img_data[:,0,:,:] -= 123.68
    img_data[:,1,:,:] -= 116.779
    img_data[:,2,:,:] -= 103.939

    img_data_1[:,0,:,:] -= 123.68
    img_data_1[:,1,:,:] -= 116.779
    img_data_1[:,2,:,:] -= 103.939

    out_file1.create_dataset("img_data",data=img_data)
    out_file1.create_dataset("img_label",data=img_label)

    out_file2.create_dataset("img_data",data=img_data_1)
    out_file2.create_dataset("img_label",data=img_label_1)
    print "\n finished writing original images"
    del img_data, img_data_1
    out_file1.close()
    out_file2.close()


    for dist_id in range(num_dist):

        img_data = np.empty((num_classes * num_val, 3, img_crop, img_crop), np.float32)
        img_data_1 = np.empty((num_classes * num_val / 2, 3, img_crop, img_crop), np.float32)
        print " \n Blur distortion level "+str(dist_id)

        out_file_blur = h5py.File('imagenet_val_blur_'+str(num_classes)+str(dist_id)+'.h5', 'w')
        out_file_blur_small = h5py.File('imagenet_dcranksubset_blur_'+str(dist_id)+'.h5', 'w')

        for class_id in range(num_classes):
            print "\n class id" + str(class_id)
            filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
            # filelist = filelist[rand_idx]
            for img_id in range(num_val):
                img_temp = cv2.imread(val_folder_loc + str(class_id) + '/'+filelist[rand_idx[img_id]])
                img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                img_temp = cv2.resize(img_temp, (img_sz, img_sz))
                img_temp = cv2.GaussianBlur(img_temp,(4*blur_std[dist_id]+1, 4*blur_std[dist_id]+1),blur_std[dist_id], None,blur_std[dist_id],cv2.BORDER_CONSTANT)
                img_temp = np.transpose(img_temp, (2, 0, 1))
                img_temp = img_temp[:, (img_sz - img_crop) // 2:(img_sz + img_crop) // 2,
                           (img_sz - img_crop) // 2:(img_sz + img_crop) // 2]
                img_data[num_val * class_id + img_id, :, :, :] = img_temp.copy()

                if img_id < 5:
                    img_data_1[num_val * class_id / 2 + img_id, :, :, :] = img_temp.copy()


        img_data[:, 0, :, :] -= 123.68
        img_data[:, 1, :, :] -= 116.779
        img_data[:, 2, :, :] -= 103.939

        img_data_1[:, 0, :, :] -= 123.68
        img_data_1[:, 1, :, :] -= 116.779
        img_data_1[:, 2, :, :] -= 103.939

        out_file_blur.create_dataset("img_data", data=img_data)
        out_file_blur.create_dataset("img_label", data=img_label)

        out_file_blur_small.create_dataset("img_data", data=img_data_1)
        out_file_blur_small.create_dataset("img_label", data=img_label_1)
        print "\n finished writing blurred images"
        out_file_blur.close()
        out_file_blur_small.close()

        del img_data, img_data_1

        img_data = np.empty((num_classes * num_val, 3, img_crop, img_crop), np.float32)
        img_data_1 = np.empty((num_classes * num_val / 2, 3, img_crop, img_crop), np.float32)
        out_file_awgn = h5py.File('imagenet_val_awgn_' +str(num_classes)+ str(dist_id) + '.h5', 'w')
        out_file_awgn_small = h5py.File('imagenet_dcranksubset_awgn_' + str(dist_id) + '.h5', 'w')

        for class_id in range(num_classes):
            print "\n class id" + str(class_id)
            filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
            # filelist = filelist[rand_idx]
            for img_id in range(num_val):
                img_temp = cv2.imread(val_folder_loc + str(class_id) + '/'+filelist[rand_idx[img_id]])
                img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                img_temp = cv2.resize(img_temp, (img_sz, img_sz))
                img_temp = img_temp.astype(np.float32)
                img_temp += np.random.normal(0, awgn_std[dist_id], (img_sz,img_sz,3))
                img_temp = np.transpose(img_temp, (2, 0, 1))
                img_temp = img_temp[:, (img_sz - img_crop) // 2:(img_sz + img_crop) // 2,
                           (img_sz - img_crop) // 2:(img_sz + img_crop) // 2]
                img_data[num_val * class_id + img_id, :, :, :] = img_temp.copy()

                if img_id < 5:
                    img_data_1[num_val * class_id / 2 + img_id, :, :, :] = img_temp.copy()


        img_data[:, 0, :, :] -= 123.68
        img_data[:, 1, :, :] -= 116.779
        img_data[:, 2, :, :] -= 103.939

        img_data_1[:, 0, :, :] -= 123.68
        img_data_1[:, 1, :, :] -= 116.779
        img_data_1[:, 2, :, :] -= 103.939

        out_file_awgn.create_dataset("img_data", data=img_data)
        out_file_awgn.create_dataset("img_label", data=img_label)

        out_file_awgn_small.create_dataset("img_data", data=img_data_1)
        out_file_awgn_small.create_dataset("img_label", data=img_label_1)
        print "\n finished writing awgn added images"
        out_file_awgn.close()
        out_file_awgn_small.close()




