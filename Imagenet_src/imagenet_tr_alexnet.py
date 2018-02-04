import threading
import numpy as np
import h5py
import cv2
import sys
from os import listdir
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras import backend as K
import time
from keras.preprocessing.image import flip_axis, transform_matrix_offset_center, apply_transform
import Imagenet_datagen as IN_dat
import alexnet_imagenet_base_acc as base_acc
import alexnet_layer_arch as DNN


# define parameters

np.random.seed(1337)
rand_idx = np.random.permutation(50)

# number of layers corrected using correction units
num_ly_corr = 5

#data augmentation parameters
rotation_range  = 20
width_shift = 0.15
height_shift = 0.15
horizontal_flip = True
zoom_range = 0.25

acc=0

# type of model : 'ft' or 'dc'
#'ft' -> finetune
#'dc' -> deepcorr
model_type = 'dc'

# type of distortion : 'blur' or 'awgn'
dist_type='blur'
num_imgs_total = 1281167

if model_type == 'dc':
    learning_rate = 0.1
else:
    learning_rate = 0.001
momentum = 0.9

# set to 1 if planning to start training from a previously stored checkpoint
start_from_old = 0

# correction unit arch: 'CW' or 'fixed' or 'bottleneck'
corr_arch ='CW'

save_epoch = 5*2

early_stop = 0

tolerance = 0.001

batch_size=200
img_per_class=2

# set to 0 for testing already trained model, else set to 50
num_epoch = 0

iter_mult = 250*IN_dat.num_classes/batch_size



# path to Imagenet training data
test_folder_loc ='../Training/Class_'

# path to Imagenet validation data
val_folder_loc = '../ILSVRC_data/Class_'

num_test = IN_dat.num_test + IN_dat.num_val

# compute accuracy for imagenet test set
def imagenet_test_eval(model, dist_type):

    batch_size = 250

    print '\n -------------------Testing on IMAGENET validation set ---------------------\n'

    ref_test_acc = np.empty((2, IN_dat.num_dist+1), np.float32)
    y_pred = np.empty((IN_dat.num_classes * num_test, IN_dat.num_classes), np.float32)
    y_label = np.empty((IN_dat.num_classes * num_test), np.float32)


    print '\n------------------Original--------------------------\n'
    for batch_id in range(IN_dat.num_classes/batch_size):
        print '\n processing batch ' + str(batch_id)
        img_data = np.empty((num_test * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
        img_label = np.empty((num_test * batch_size), np.float32)
        for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

            filelist = listdir(str(val_folder_loc +str(class_id) + '/'))
            for img_id in range(num_test):
                img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                img_temp = np.transpose(img_temp, (2, 0, 1))
                img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2,
                           (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                img_data[num_test * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                img_label[num_test * (class_id - batch_size * batch_id - 1) + img_id] = class_id

        img_data[:, 0, :, :] -= 123.68
        img_data[:, 1, :, :] -= 116.779
        img_data[:, 2, :, :] -= 103.939

        y_pred[batch_id * num_test * batch_size:(batch_id + 1) * num_test * batch_size, :] = model.predict(img_data,batch_size=250)
        y_label[batch_size * num_test * batch_id:(batch_id + 1) * num_test * batch_size] = img_label

        del img_data, img_label



    ref_test_acc[0, 0], ref_test_acc[1, 0] = base_acc.compute_test_accuracy(y_pred, y_label)

    del y_pred, y_label


    if dist_type=='blur':
        for dist_id in range(IN_dat.num_dist):
            print '\n ----------------------Blur/AWGN distortion level ' + str(dist_id) + '----------------------\n'

            y_pred = np.empty((IN_dat.num_classes * num_test, IN_dat.num_classes), np.float32)
            y_label = np.empty((IN_dat.num_classes * num_test), np.float32)

            batch_size = 250
            for batch_id in range(IN_dat.num_classes/batch_size):
                print '\n processing batch ' + str(batch_id)
                img_data = np.empty((num_test * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
                img_label = np.empty((num_test * batch_size), np.float32)
                for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

                    filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
                    for img_id in range(num_test):
                        img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                        img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                        img_temp = cv2.GaussianBlur(img_temp, (4 * IN_dat.blur_std[dist_id] + 1, 4 * IN_dat.blur_std[dist_id] + 1),
                                                    IN_dat.blur_std[dist_id], None, IN_dat.blur_std[dist_id], cv2.BORDER_CONSTANT)
                        img_temp = np.transpose(img_temp, (2, 0, 1))
                        img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop)
                                                // 2,(IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                        img_data[num_test * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                        img_label[num_test * (class_id - batch_size * batch_id - 1) + img_id] = class_id

                img_data[:, 0, :, :] -= 123.68
                img_data[:, 1, :, :] -= 116.779
                img_data[:, 2, :, :] -= 103.939


                y_pred[batch_id * num_test * batch_size:(batch_id + 1) * num_test * batch_size, :] = model.predict(img_data, batch_size=250)
                y_label[batch_size * num_test * batch_id:(batch_id + 1) * num_test * batch_size] = img_label

                del img_data, img_label

            ref_test_acc[0, dist_id+1], ref_test_acc[1, dist_id+1] = base_acc.compute_test_accuracy(y_pred, y_label)

            del y_pred, y_label
    else:
        for dist_id in range(IN_dat.num_dist):
            print '\n ----------------------Blur/AWGN distortion level ' + str(dist_id) + '----------------------\n'


            y_pred = np.empty((IN_dat.num_classes * num_test, IN_dat.num_classes), np.float32)
            y_label = np.empty((IN_dat.num_classes * num_test), np.float32)

            batch_size = 250
            for batch_id in range(IN_dat.num_classes/batch_size):
                print '\n processing batch ' + str(batch_id)
                img_data = np.empty((num_test * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
                img_label = np.empty((num_test * batch_size), np.float32)
                for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

                    filelist = listdir(str(val_folder_loc + str(class_id) + '/'))

                    for img_id in range(num_test):
                        img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                        img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                        img_temp = img_temp.astype(np.float32)
                        img_temp += np.random.normal(0, IN_dat.awgn_std[dist_id], (IN_dat.img_sz, IN_dat.img_sz, 3))
                        img_temp = np.transpose(img_temp, (2, 0, 1))
                        img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop)
                                        // 2,(IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                        img_data[num_test * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                        img_label[num_test * (class_id - batch_size * batch_id - 1) + img_id] = class_id

                img_data[:, 0, :, :] -= 123.68
                img_data[:, 1, :, :] -= 116.779
                img_data[:, 2, :, :] -= 103.939

                y_pred[batch_id * num_test * batch_size:(batch_id + 1) * num_test * batch_size, :] = model.predict(img_data, batch_size=250)
                y_label[batch_size * num_test * batch_id:(batch_id + 1) * num_test * batch_size] = img_label

                del img_data, img_label

            ref_test_acc[0, dist_id+1], ref_test_acc[1, dist_id+1] = base_acc.compute_test_accuracy(y_pred, y_label)

    print'\n-------------------Top 1 acc ---------------------------------\n'
    print ref_test_acc[0, :]
    print '\n ---------------------Top 5 acc--------------------------------\n'
    print ref_test_acc[1, :]

    # out_file = h5py.File('alexnet_imagenet_correction_' + str(IN_dat.num_classes) + '_' + str(dist_type) + '_acc.h5', 'w')
    #
    # out_file.create_dataset('test_acc', data=ref_test_acc)
    # out_file.close()


# compute accuracy for validation set
def compute_validation_acc(model,dist_type='blur'):

    batch_size = 250
    num_val = 10
    print '\n -------------------Testing on IMAGENET validation set ---------------------\n'

    ref_test_acc = np.empty((2, IN_dat.num_dist+1), np.float32)
    y_pred = np.empty((IN_dat.num_classes * num_val, IN_dat.num_classes), np.float32)
    y_label = np.empty((IN_dat.num_classes * num_val), np.float32)


    print '\n------------------Original--------------------------\n'
    for batch_id in range(IN_dat.num_classes/batch_size):
        print '\n processing batch ' + str(batch_id)
        img_data = np.empty((num_val * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
        img_label = np.empty((num_val * batch_size), np.float32)
        for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

            filelist = listdir(str(val_folder_loc +str(class_id) + '/'))
            for img_id in range(num_val):
                img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                img_temp = np.transpose(img_temp, (2, 0, 1))
                img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2,
                           (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                img_data[num_val * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                img_label[num_val * (class_id - batch_size * batch_id - 1) + img_id] = class_id

        img_data[:, 0, :, :] -= 123.68
        img_data[:, 1, :, :] -= 116.779
        img_data[:, 2, :, :] -= 103.939

        y_pred[batch_id * num_val * batch_size:(batch_id + 1) * num_val * batch_size, :] = model.predict(img_data,batch_size=250)
        y_label[batch_size * num_val * batch_id:(batch_id + 1) * num_val * batch_size] = img_label

        del img_data, img_label



    ref_test_acc[0, 0], ref_test_acc[1, 0] = base_acc.compute_test_accuracy(y_pred, y_label)

    del y_pred, y_label


    if dist_type=='blur':
        for dist_id in range(IN_dat.num_dist):
            print '\n ----------------------Blur/AWGN distortion level ' + str(dist_id) + '----------------------\n'

            y_pred = np.empty((IN_dat.num_classes * num_val, IN_dat.num_classes), np.float32)
            y_label = np.empty((IN_dat.num_classes * num_val), np.float32)

            batch_size = 250
            for batch_id in range(IN_dat.num_classes/batch_size):
                print '\n processing batch ' + str(batch_id)
                img_data = np.empty((num_val * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
                img_label = np.empty((num_val * batch_size), np.float32)
                for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

                    filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
                    for img_id in range(num_val):
                        img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                        img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                        img_temp = cv2.GaussianBlur(img_temp, (4 * IN_dat.blur_std[dist_id] + 1, 4 * IN_dat.blur_std[dist_id] + 1),
                                                    IN_dat.blur_std[dist_id], None, IN_dat.blur_std[dist_id], cv2.BORDER_CONSTANT)
                        img_temp = np.transpose(img_temp, (2, 0, 1))
                        img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop)
                                                // 2,(IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                        img_data[num_val * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                        img_label[num_val * (class_id - batch_size * batch_id - 1) + img_id] = class_id

                img_data[:, 0, :, :] -= 123.68
                img_data[:, 1, :, :] -= 116.779
                img_data[:, 2, :, :] -= 103.939


                y_pred[batch_id * num_val * batch_size:(batch_id + 1) * num_val * batch_size, :] = model.predict(img_data, batch_size=250)
                y_label[batch_size * num_val * batch_id:(batch_id + 1) * num_val * batch_size] = img_label

                del img_data, img_label

            ref_test_acc[0, dist_id+1], ref_test_acc[1, dist_id+1] = base_acc.compute_test_accuracy(y_pred, y_label)

            del y_pred, y_label
    else:
        for dist_id in range(IN_dat.num_dist):
            print '\n ----------------------Blur/AWGN distortion level ' + str(dist_id) + '----------------------\n'


            y_pred = np.empty((IN_dat.num_classes * num_val, IN_dat.num_classes), np.float32)
            y_label = np.empty((IN_dat.num_classes * num_val), np.float32)

            batch_size = 250
            for batch_id in range(IN_dat.num_classes/batch_size):
                print '\n processing batch ' + str(batch_id)
                img_data = np.empty((num_val * batch_size, 3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
                img_label = np.empty((num_val * batch_size), np.float32)
                for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

                    filelist = listdir(str(val_folder_loc + str(class_id) + '/'))

                    for img_id in range(num_val):
                        img_temp = cv2.imread(val_folder_loc + str(class_id) + '/' + filelist[rand_idx[img_id]])
                        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                        img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
                        img_temp = img_temp.astype(np.float32)
                        img_temp += np.random.normal(0, IN_dat.awgn_std[dist_id], (IN_dat.img_sz, IN_dat.img_sz, 3))
                        img_temp = np.transpose(img_temp, (2, 0, 1))
                        img_temp = img_temp[:, (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop)
                                        // 2,(IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2]
                        img_data[num_val * (class_id - batch_size * batch_id - 1) + img_id, :, :, :] = img_temp.copy()
                        img_label[num_val * (class_id - batch_size * batch_id - 1) + img_id] = class_id

                img_data[:, 0, :, :] -= 123.68
                img_data[:, 1, :, :] -= 116.779
                img_data[:, 2, :, :] -= 103.939

                y_pred[batch_id * num_val * batch_size:(batch_id + 1) * num_val * batch_size, :] = model.predict(img_data, batch_size=250)
                y_label[batch_size * num_val * batch_id:(batch_id + 1) * num_val * batch_size] = img_label

                del img_data, img_label

            ref_test_acc[0, dist_id+1], ref_test_acc[1, dist_id+1] = base_acc.compute_test_accuracy(y_pred, y_label)

    print'\n-------------------Top 1 acc ---------------------------------\n'
    print ref_test_acc[0, :]
    print '\n ---------------------Top 5 acc--------------------------------\n'
    print ref_test_acc[1, :]

    return np.mean(ref_test_acc[0,:]), np.mean(ref_test_acc[1,:])



# read input image perform pre-processing operations
def get_inp_img(filename):
    img_temp = cv2.imread(filename)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
    img_temp = cv2.resize(img_temp, (IN_dat.img_sz, IN_dat.img_sz))
    # img_temp = np.transpose(img_temp, (2, 0, 1))
    img_temp = img_temp[(IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2,
               (IN_dat.img_sz - IN_dat.img_crop) // 2:(IN_dat.img_sz + IN_dat.img_crop) // 2,:]
    return img_temp

# data augmentation methods
def get_random_transform(x,rotation_range,shift_range,zoom,horizontal_flip):
    """Randomly augment a single image tensor.
    # Arguments
        x: 3D tensor, single image.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0
    channel_axis = 3
    row_axis = 1
    col_axis = 2
    img_row_axis = row_axis - 1
    img_col_axis = col_axis - 1
    img_channel_axis = channel_axis - 1

    zoom_range = [1 - zoom, 1 + zoom]
    fill_mode = 'nearest'
    cval = 0.
    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0

    if shift_range:
        tx = np.random.uniform(-shift_range, shift_range) * x.shape[img_row_axis]
    else:
        tx = 0

    if shift_range:
        ty = np.random.uniform(-shift_range, shift_range) * x.shape[img_col_axis]
    else:
        ty = 0

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)


    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=fill_mode, cval=cval)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)

    return x

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
          return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
#



@threadsafe_generator
def mygenerator(batch_size,img_per_class):
    total_imgs = 1250
    # class_numbers = np.random.permutation(num_classes)
    class_numbers=np.arange(IN_dat.num_classes)
    while(1):
        class_numbers= class_numbers[np.random.permutation(IN_dat.num_classes)]

        for num_imgs in range(total_imgs*IN_dat.num_classes/batch_size):

            batch_id = num_imgs%(IN_dat.num_classes*img_per_class/batch_size)
            xbatch = np.empty((batch_size,3, IN_dat.img_crop, IN_dat.img_crop), np.float32)
            ybatch = np.empty((batch_size),np.float32)
            sample_count = 0

            for class_id in range(batch_id*(batch_size/img_per_class),(batch_id+1)*(batch_size/img_per_class)):
                filelist = listdir(str(test_folder_loc + str(class_numbers[class_id]) + '/'))
                file_ids = np.random.randint(0,len(filelist),img_per_class)
                for img_id in range(img_per_class):
                    fname = test_folder_loc + str(class_numbers[class_id]) + '/' + filelist[file_ids[img_id]]
                    inp_img = get_inp_img(fname)
                    if np.random.random()>0.5:
                        dist_lvl = np.random.randint(0, IN_dat.num_dist, 1)
                        if dist_type=='blur':

                            inp_img = cv2.GaussianBlur(inp_img, (4 * IN_dat.blur_std[dist_lvl[0]] + 1, 4 * IN_dat.blur_std[dist_lvl[0]] + 1),
                                            IN_dat.blur_std[dist_lvl[0]], None, IN_dat.blur_std[dist_lvl[0]], cv2.BORDER_CONSTANT)
                        else:
                            inp_img  = inp_img.astype(np.float32)
                            inp_img += np.random.normal(0, IN_dat.awgn_std[dist_lvl[0]], (IN_dat.img_crop, IN_dat.img_crop, 3))
                    else:
                        if np.random.random()>0.5:
                            inp_img = get_random_transform(inp_img,rotation_range,width_shift,zoom_range,horizontal_flip)


                    inp_img =np.transpose(inp_img,(2,0,1))
                    xbatch[sample_count,:,:,:] = inp_img.copy()
                    ybatch[sample_count] = class_numbers[class_id]
                    sample_count += 1

            xbatch[:,0,:,:] -= 123.68
            xbatch[:, 1, :, :] -= 116.779
            xbatch[:, 2, :, :] -= 103.939

            ybatch = to_categorical(ybatch,IN_dat.num_classes)
            yield xbatch,ybatch


if start_from_old==1:
    if model_type=='dc':
        model, model_dict = DNN.alexnet_correct('../Imagenet_models/alexnet_imagenet_weights.npy',out_dim=IN_dat.num_classes,dist_type=dist_type,corr_arch=corr_arch,num_ly_corr=num_ly_corr)
        tr_wts = np.load('../Imagenet_models/alexnet_imagenet_current_state' + str(IN_dat.num_classes) + '_' + str(dist_type) + '_' + str(num_ly_corr) + '_'+str(corr_arch)+'.npy')
    else:
        model, model_dict =  DNN.AlexNetDNN('../Imagenet_models/alexnet_weights.h5',heatmap=False,trainable=True,out_dim=IN_dat.num_classes)
        tr_wts = np.load('../Imagenet_models/alexnet_imagenet_ft_current_state' + str(IN_dat.num_classes) + '_' + str(dist_type) + '.npy')
    model.set_weights(tr_wts)
    best_val_acc = 0.
    start_epoch = 0
    win_epoch =0
    inc = 1
else:
    if model_type=='dc':
        model, model_dict = DNN.alexnet_correct('../Imagenet_models/alexnet_imagenet_weights.npy',out_dim=IN_dat.num_classes,dist_type=dist_type,corr_arch=corr_arch,num_ly_corr=num_ly_corr)
    else:
        model, model_dict = DNN.AlexNetDNN('../Imagenet_models/alexnet_weights.h5',heatmap=False,trainable=True,out_dim=IN_dat.num_classes)
    best_val_acc = 0
    start_epoch = 0
    win_epoch = 0
    inc = 1


sgd = SGD(lr=learning_rate, momentum=momentum, decay=0.0, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print ('compiled model successfully')


# print model.summary()


source_gen  = mygenerator(batch_size,img_per_class)

# training iterations
for e in range(start_epoch,num_epoch):

    print '--------------------------------------------------------------Iteration :--------------------------------------------------------------------- \n' + str((e + 1))
    print 'Learning rate : ' + str(K.get_value(model.optimizer.lr))
    t1 = time.time()
    model.fit_generator(source_gen,iter_mult*batch_size,1,verbose=1,nb_worker=10)
#

    val_acc_1, val_acc_5 = compute_validation_acc(model,dist_type)



    print'\n Validation top 1 acc : ' + str(val_acc_1)
    print'\n Validation top 5 acc : ' + str(val_acc_5)
    print'-------------------------------------------------------------------------------------------------------------------------------------------'

    # save best validating model
    if val_acc_1 >= best_val_acc + tolerance:
        print' \n Found new model at iteration : ' + str((e + 1)*iter_mult)
        best_val_acc = val_acc_1
        tr_wts = model.get_weights()
        if model_type=='dc':
            np.save('../Imagenet_models/alexnet_imagenet_corr_'+str(IN_dat.num_classes)+'_'+str(dist_type)+'_'+str(num_ly_corr)+'_'+str(corr_arch)+'.npy', tr_wts)
        else:
            np.save('../Imagenet_models/alexnet_imagenet_ft_'+str(IN_dat.num_classes)+'_'+str(dist_type)+'.npy', tr_wts)
        win_epoch = e + 1
        inc = 1


    if (e+1)%15==0:
        print '\n reducing learning rate by 10'
        l_r = K.get_value(model.optimizer.lr) * 0.1
        tolerance = tolerance/2
        K.set_value(model.optimizer.lr, l_r)
        inc += 1


    # write out training log
    if model_type=='dc':
        with open("alexnet_imagenet_training_stat" + str(IN_dat.num_classes) + "_" + str(dist_type) + "_" + str(num_ly_corr) + "_" + str(corr_arch) + ".txt", 'a') as myFile:
            myFile.write('\n Iteration : ' + str((e + 1) * iter_mult) + ' Learning rate : ' + str(K.get_value(model.optimizer.lr))
                         + ' Validation acc top1  : ' + str(val_acc_1) + ' Validation acc top5 : ' + str(val_acc_5))
    else:
        with open("alexnet_imagenet_training_stat" + str(IN_dat.num_classes) + ".txt", 'a') as myFile:
            myFile.write('\n Iteration : ' + str((e + 1) * iter_mult) + ' Learning rate : ' + str(K.get_value(model.optimizer.lr))
                         + ' Validation acc top1  : ' + str(val_acc_1) + ' Validation acc top5 : ' + str(val_acc_5))

    current_wts = model.get_weights()

    # save current weights to start from in case code crash or training termination
    if model_type=='dc':
        np.save('../Imagenet_models/alexnet_imagenet_current_state' + str(IN_dat.num_classes) + '_' + str(dist_type) + '_' + str(num_ly_corr) + '_'+str(corr_arch)+'.npy', current_wts)
    else:
        np.save('../Imagenet_models/alexnet_imagenet_ft_current_state' + str(IN_dat.num_classes) + '_' + str(dist_type) + '.npy', current_wts)


# load model weights for testing
if model_type=='dc':
    final_wts = np.load('../Imagenet_models/alexnet_imagenet_corr_'+str(IN_dat.num_classes)+'_'+str(dist_type)+'_'+str(num_ly_corr)+'_'+str(corr_arch)+'.npy')
else:
    final_wts = np.load('../Imagenet_models/alexnet_imagenet_ft_'+str(IN_dat.num_classes)+'_'+str(dist_type)+'.npy')

model.set_weights(final_wts)

print dist_type
print model_type
# print IN_dat.blur_std


# test trained model
imagenet_test_eval(model,dist_type=dist_type)



