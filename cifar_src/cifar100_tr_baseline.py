import numpy as np
np.random.seed(1337)


from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import cifar_layer_arch as CIFAR_arch

#load CIFAR-100 dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
perm_ids = np.random.permutation(len(X_train))
out_dim  = 100

# subtract mean and normalize with std dev. of data
def normalize_data(X_train, X_test):
    X_train = X_train.astype(float, copy=False)
    X_test = X_test.astype(float, copy=False)

    mean_r = np.mean(X_train[:, 0, :, :])
    mean_g = np.mean(X_train[:, 1, :, :])
    mean_b = np.mean(X_train[:, 2, :, :])

    std_r = np.std(X_train[:, 0, :, :])
    std_g = np.std(X_train[:, 1, :, :])
    std_b = np.std(X_train[:, 2, :, :])


    X_train[:, 0, :, :] -= mean_r
    X_train[:, 1, :, :] -= mean_g
    X_train[:, 2, :, :] -= mean_b

    X_train[:, 0, :, :] /= std_r
    X_train[:, 1, :, :] /= std_g
    X_train[:, 2, :, :] /= std_b

    X_train = X_train[perm_ids, :, :, :]

    X_test[:, 0, :, :] -= mean_r
    X_test[:, 1, :, :] -= mean_g
    X_test[:, 2, :, :] -= mean_b

    X_test[:, 0, :, :] /= std_r
    X_test[:, 1, :, :] /= std_g
    X_test[:, 2, :, :] /= std_b

    return X_train, X_test


X_train, X_test = normalize_data(X_train, X_test)

Y_train = to_categorical(y_train,nb_classes=out_dim)
Y_train = Y_train[perm_ids]
Y_test = to_categorical(y_test,nb_classes=out_dim)



datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.15,height_shift_range=0.15,horizontal_flip=True)

model, layer_dict = CIFAR_arch.cifar10_100()

# set training parameters
num_epoch = 110
learning_rate = 0.1
momentum = 0.9


def compute_test_accuracy(model,x_test, y_test):
    y_pred = model.predict(x_test,batch_size=128)

    top1_err_drop = 0

    for test_id in range(len(y_pred)):

        if(np.argmax(y_pred[test_id,:])!= np.argmax(y_test[test_id,:])):
            top1_err_drop += 1

    accuracy = 1-float(top1_err_drop)/float(len(y_test))
    print '\n Accuracy is : '+str(accuracy)
    return accuracy
    
   
sgd = SGD(lr=learning_rate,momentum=momentum,decay=0,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
print ('compiled model successfully')

num_train = int(np.ceil(X_train.shape[0]*0.9))
best_val_acc = 0 
win_epoch  = 0
patience =  30
early_stop = 0
best_tr_acc = 0
tolerance = 0.001


for e in range(num_epoch):
    
    print 'Epoch : \n'+str(e+1)
    print 'Learning rate : '+str(K.get_value(model.optimizer.lr))

    model.fit_generator(datagen.flow(X_train[0:num_train,:,:,:],Y_train[0:num_train],batch_size=256),samples_per_epoch=len(X_train[0:num_train,:,:,:]),nb_epoch=1,verbose=1)

    tr_acc = compute_test_accuracy(model,X_train[0:num_train,:,:,:],Y_train[0:num_train])
    val_acc = compute_test_accuracy(model,X_train[num_train:len(X_train),:,:,:],Y_train[num_train:len(X_train)]) 
    print ' Training acc : '+str(tr_acc) +'\t Validation acc : '+str(val_acc)
    if np.mod(e+1,30)==0:
        print '\n reducing learning rate by half'
        l_r = K.get_value(model.optimizer.lr)*0.1
        K.set_value(model.optimizer.lr,l_r)

    if val_acc >= best_val_acc+tolerance:
        print' \n Found new model at epoch : '+str(e+1)
        best_val_acc = val_acc 
        tr_wts = model.get_weights()
        np.save('../cifar_models/CIFAR_100_fine_best_model.npy',tr_wts)


    if early_stop == 1:
        print'\n stopping early at epoch : '+str(e+1)
        break



print '\n Testing : '
final_wts = np.load('../cifar_models/CIFAR_100_fine_best_model.npy')
model.set_weights(final_wts) 
test_acc = compute_test_accuracy(model,X_test,Y_test)

