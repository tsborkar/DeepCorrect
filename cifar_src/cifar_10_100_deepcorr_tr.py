import numpy as np
import sys



np.random.seed(1337)
import h5py
import cv2

from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.datasets import cifar100

import cifar_layer_arch as CIFAR
import cifar_distortions as CIFAR_acc

#set distortion type
dist_type ='blur'

# set type of model : 'ft' for finetune or 'dc' for deepcorr
model_type ='dc'

# set number of layers corrected or finetuned : 1 < num_ly_corr <= 6
num_ly_corr = 1

# set fraction of filters corrected per layer : 0 < correction_perc < 1
correction_perc = 0.5

# Performs training followed by testing when num_epoch > 0, while num_epoch = 0 performs only testing
num_epoch = 0


momentum = 0.9
num_dist = 6
img_width, img_height = 32, 32
nchn = 3
out_dim = 100
inp_chn  =  3
img_w = 32
blur_std =[0.5,1.,1.5,2.,2.5,3.]
blur_win = [3,5,7,9,11,13]
awgn_std =[5,10,15,20,25,30]
best_val_acc = 0
win_epoch = 0
patience = 20
early_stop = 0
best_tr_acc = 0
tolerance = 0.001
batch_size = 250
total_count = 0
inc = 1


(X_train1, y_train1), (X_test, y_test) = cifar100.load_data(label_mode='fine')

X_train1 = X_train1.astype(float,copy=False)
X_test = X_test.astype(float,copy=False)
np.random.seed(1337)
perm_ids = np.random.permutation(len(X_train1))
perm_id = np.random.permutation(6)








if model_type=="ft":
    model, model_dict = CIFAR.cifar10_100('../cifar_models/CIFAR_100_fine_best_model.npy',out_dim=100,num_ly_corr = num_ly_corr)
    learning_rate = 0.01
    print '\n Finetune model with number of trainable layers '+str(num_ly_corr)

else:
    print correction_perc
    model = CIFAR.get_correct_net('../cifar_models/CIFAR_100_fine_best_model.npy',corr_lvl=num_ly_corr,dist_type = dist_type, correction_perc=correction_perc)
    if num_ly_corr>4:
        learning_rate = 0.01
    else:
        learning_rate = 0.1
    # print '\n learn correction unit'




mean_r = np.mean(X_train1[:,0,:,:])
mean_g = np.mean(X_train1[:,1,:,:])
mean_b = np.mean(X_train1[:,2,:,:])

std_r = np.std(X_train1[:,0,:,:])
std_g = np.std(X_train1[:,1,:,:])
std_b = np.std(X_train1[:,2,:,:])

X_train1 = X_train1[perm_ids,:,:,:]



#

y_train1 = y_train1[perm_ids]



Y_test =  to_categorical(y_test,nb_classes=out_dim)


sgd = SGD(lr=learning_rate, momentum=momentum,decay=1e-5, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['acc'])
print ('compiled model successfully')
model.summary()

num_train = int(np.ceil(X_train1.shape[0]*0.9))
num_val = int(np.ceil(X_train1.shape[0]*0.1))


X_train = np.empty((num_train,3,img_width,img_height),np.float32)
y_train = np.empty((num_train,1),int)
X_val = np.empty(((num_dist+1)*num_val,3,img_width,img_height),np.float32)
y_val = np.empty(((num_dist+1)*num_val,1), int)
X_train = X_train1[0:num_train,:,:,:].copy()
X_val[0:num_val,:,:,:] = X_train1[num_train:num_train+num_val,:,:,:].copy()
y_train = y_train1[0:num_train].copy()
y_val[0:num_val] = y_train1[num_train:num_train+num_val].copy()
del X_train1, y_train1




for iter_id in range(num_dist):
    # print str(iter_id) + '\t '
    for img_id in range(num_val):

        temp_img = X_val[img_id,:,:,:].copy()
        if dist_type=='blur':
            temp_img = np.transpose(temp_img,(1,2,0))

            temp_img = cv2.GaussianBlur(temp_img,(blur_win[iter_id],blur_win[iter_id]),blur_std[iter_id],None,blur_std[iter_id],cv2.BORDER_CONSTANT)
            temp_img = np.transpose(temp_img,(2,0,1))
        else:
            temp_img += np.random.normal(0, awgn_std[iter_id], (3, img_width, img_height))

        temp_label = y_val[img_id].copy()
        X_val[(iter_id + 1) * num_val + img_id, :, :, :] = temp_img
        y_val[(iter_id + 1) * num_val + img_id] = temp_label

        del temp_img, temp_label


Y_train = to_categorical(y_train,out_dim)
del y_train
Y_val = to_categorical(y_val,out_dim)
del y_val


X_val[:,0,:,:] -= mean_r
X_val[:,1,:,:] -= mean_g
X_val[:,2,:,:] -= mean_b

X_val[:,0,:,:] /= std_r
X_val[:,1,:,:] /= std_g
X_val[:,2,:,:] /= std_b


datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True)

def mygenerator(datagen,xtrain,ytrain, batch_sz):

    for x_batch, y_batch in datagen.flow(xtrain, ytrain, batch_size=batch_sz):
        dist_indx = np.random.randint(low=0, high=6, size=x_batch.shape[0] / 2)
        assert len(dist_indx) == x_batch.shape[0] / 2
        for i in range(batch_sz / 2):
            if dist_type=='blur':
                x_batch[batch_sz / 2 + i, :, :, :] = np.transpose(cv2.GaussianBlur(np.transpose(x_batch[batch_sz / 2 + i, :, :, :], (1, 2, 0)),
                                        (blur_win[dist_indx[i]], blur_win[dist_indx[i]]), blur_std[dist_indx[i]], None, blur_std[dist_indx[i]],
                                 cv2.BORDER_CONSTANT), (2, 0, 1))
            else:
                x_batch[batch_sz / 2 + i, :, :, :] += np.random.normal(0,awgn_std[dist_indx[i]],(3,img_width,img_height))

        x_batch[:, 0, :, :] -= mean_r
        x_batch[:, 1, :, :] -= mean_g
        x_batch[:, 2, :, :] -= mean_b
        x_batch[:,0,:,:] /= std_r
        x_batch[:,1,:,:] /= std_g
        x_batch[:,2,:,:] /= std_b
        yield x_batch, y_batch

source_gen = mygenerator(datagen,X_train,Y_train,batch_sz=batch_size)


iter_mult = 360
print '\n distortion type '+str(dist_type)
for e in range(num_epoch):

    print 'Epoch : \n' + str(e + 1)
    print 'Learning rate : ' + str(K.get_value(model.optimizer.lr))


    model.fit_generator(source_gen,samples_per_epoch=2*X_train.shape[0],verbose=1,nb_epoch=1)

    val_acc = CIFAR_acc.compute_test_accuracy(model, X_val, Y_val)
    print' Validation acc : ' + str(val_acc)

    if val_acc >= best_val_acc + tolerance:
        print' \n Found new model at epoch : ' + str(e + 1)
        best_val_acc = val_acc
        tr_wts = model.get_weights()
        if model_type=="ft":
            np.save('../cifar_models/CIFAR_100_'+str(dist_type)+'_finetune_layers_'+str(num_ly_corr)+'.npy', tr_wts)
        
        else:

            np.save('../cifar_models/CIFAR_100_'+str(dist_type)+'_'+model_type+'_top'+str(int(100*correction_perc))+'_l'+str(num_ly_corr)+'.npy', tr_wts)
        win_epoch = e + 1
        inc = 1

    if model_type=='ft':
        with open("cifar100_training_stat_"+str(dist_type)+"_"+str(num_ly_corr)+"_finetune.txt",'a') as myFile:
            myFile.write('\n Iteration : '+str((e+1)*iter_mult)+' Learning rate : '+str(K.get_value(model.optimizer.lr))+' Validation acc top1  : '+str(val_acc))
#
    else:

        with open("cifar100_training_stat_"+str(dist_type)+'_'+model_type+"_"+str(num_ly_corr)+".txt",'a') as myFile:
            myFile.write('\n Iteration : '+str((e+1)*iter_mult)+' Learning rate : '+str(K.get_value(model.optimizer.lr))+' Validation acc top1  : '+str(val_acc))

    if model_type=='dc':
        if e+1==1 and num_ly_corr>4:
            l_r = K.get_value(model.optimizer.lr) *10
            K.set_value(model.optimizer.lr, l_r)
       
    if np.mod(e+1,10)==0:
        print '\n reducing learning rate by 10'
        l_r = K.get_value(model.optimizer.lr) * 0.1
        K.set_value(model.optimizer.lr, l_r)
        inc += 1

print '\n Testing : '

if model_type=="ft":
    final_wts = np.load('../cifar_models/CIFAR_100_'+str(dist_type)+'_finetune_layers_'+str(num_ly_corr)+'.npy')

else:

    final_wts = np.load('../cifar_models/CIFAR_100_'+str(dist_type)+'_'+model_type+'_top'+str(int(100*correction_perc))+'_l'+str(num_ly_corr)+'.npy')

model.set_weights(final_wts)
test_acc = np.empty(7,np.float32)

# print X_test.shape
# print Y_test.shape
# raw_input('press enter to continue ')
#

X_test1 = X_test.copy()
X_test1[:,0,:,:] -= mean_r
X_test1[:,1,:,:] -= mean_g
X_test1[:,2,:,:] -= mean_b

X_test1[:,0,:,:] /= std_r
X_test1[:,1,:,:] /= std_g
X_test1[:,2,:,:] /= std_b
test_acc[0]= CIFAR_acc.compute_test_accuracy(model, X_test1, Y_test)
del X_test1

for iter_id in range(6):
    X_test1 = np.empty((Y_test.shape[0], 3, img_width, img_height), np.float32)
    for samp_id in range(Y_test.shape[0]):
        img_temp = np.copy(X_test[samp_id,:])
        if dist_type=='blur':
            img_temp =np.transpose(img_temp,(1, 2, 0))
            img_temp = cv2.GaussianBlur(img_temp, (blur_win[iter_id], blur_win[iter_id]), blur_std[iter_id], None,blur_std[iter_id], cv2.BORDER_CONSTANT)
            img_temp = np.transpose(img_temp,(2, 0, 1))
        else:
            img_temp = img_temp + np.random.normal(0,awgn_std[iter_id],(3,img_width,img_height))
        X_test1[samp_id,:] = np.copy(img_temp)
        del img_temp

    X_test1[:,0,:,:] -= mean_r
    X_test1[:,1,:,:] -= mean_g
    X_test1[:,2,:,:] -= mean_b

    X_test1[:,0,:,:] /= std_r
    X_test1[:,1,:,:] /= std_g
    X_test1[:,2,:,:] /= std_b
    test_acc[1+iter_id] = CIFAR_acc.compute_test_accuracy(model, X_test1,Y_test)
    del X_test1

#
if model_type=="ft":
    res_file = h5py.File("cifar100_"+str(dist_type)+"_num_layers_corr_"+str(num_ly_corr)+"_fine_tune_results.h5",'w')
    res_file.create_dataset('test_acc', data=test_acc)
else:

    res_file = h5py.File("cifar100_"+str(dist_type)+'_'+model_type+"_num_layers_corr_"+str(num_ly_corr)+'_beta_'+str(int(100*correction_perc))+".h5",'w')
    res_file.create_dataset('test_acc', data=test_acc)

res_file.close()
