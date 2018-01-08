
import numpy as np
import h5py
import time
from keras.optimizers import SGD

import alexnet_layer_arch as alexnet
import alexnet_imagenet_base_acc as base_acc
import Imagenet_datagen as IN_dat
import alexnet_imagenet_layers as DNN_layers
import matplotlib.pyplot as plt


# filter_ranked : 0 or 1
# set to 0 for recomputing ranking priorities
# set to 1 for using precomputed ranking priorities
filter_ranked  = 1
num_test = 5000
out_dim  = 1000
batch_size = 1000
filt_steps =[0.1, 0.25, 0.5, 0.75, 0.9]

def get_layer_swap_acc():
    num_val = 5
    out_dim = IN_dat.num_classes
    batch_size  = num_val*IN_dat.num_classes/5
    np.set_printoptions(precision=6,suppress=True)

    print "----------------imagenet---------------------------"

    inp_file = h5py.File('imagenet_dcranksubset_ori.h5','r')
    yval = inp_file['img_label'][:]
    inp_file.close()

    for layer_id in range(0,len(alexnet.layer_names),1):

        out_file2 = h5py.File('alexnet_imagenet_layer_swap_acc.h5', 'a')

        model100_l1, layer_dict100_l1 = alexnet.AlexNetDNN_layers('../Imagenet_models/alexnet_imagenet_weights.npy',layer_id=layer_id,trainable=False,out_dim=IN_dat.num_classes)
        model100_l1.compile(optimizer=SGD(), loss='categorical_crossentropy')

        print '\n compiled model successfully '
        print '\n layer name : ',alexnet.layer_names[layer_id]
        acc_dist = np.empty((2,alexnet.layer_size101[layer_id,0], IN_dat.num_dist), np.float32)
        acc_blur = np.empty((2,alexnet.layer_size101[layer_id,0], IN_dat.num_dist), np.float32)

        for iter_id in range(IN_dat.num_dist):

            vgg_ori = h5py.File('alexnet_imagenet_layer_outputs.h5', 'r')
            vgg_dist = h5py.File('alexnet_imagenet_layer_outputs_awgn_' + str(iter_id) + '.h5', 'r')
            vgg_blur = h5py.File('alexnet_imagenet_layer_outputs_blur_' + str(iter_id) + '.h5', 'r')



            y_pred_dist = np.empty((alexnet.layer_size101[layer_id, 0], num_val*IN_dat.num_classes, out_dim), np.float32)
            y_pred_blur = np.empty((alexnet.layer_size101[layer_id, 0], num_val*IN_dat.num_classes, out_dim), np.float32)




            for batch_id in range(0, (num_val*IN_dat.num_classes) / batch_size):
                print '\n Processing batch : ' + str(batch_id + 1)
                t1 = time.time()

                vgg_ori_batch1 = vgg_ori[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()


                vgg_dist_batch = vgg_dist[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()


                vgg_blur_batch = vgg_blur[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()


                t2 = time.time()
                print '\n time required for loading file ' + str(t2 - t1)

                for filter_id in range(alexnet.layer_size101[layer_id, 0]):

                    vgg_dist_t = vgg_dist_batch.copy()
                    vgg_blur_t = vgg_blur_batch.copy()

                    t3 = time.time()

                    vgg_dist_t[:, filter_id, :, :] = vgg_ori_batch1[:, filter_id, :, :].copy()

                    y_pred_dist[filter_id, batch_id * batch_size:(batch_id + 1) * batch_size, :] = model100_l1.predict(vgg_dist_t)
                    # del vgg_ori_batch2

                    vgg_blur_t[:, filter_id, :, :] = vgg_ori_batch1[:, filter_id, :, :].copy()

                    y_pred_blur[filter_id, batch_id * batch_size:(batch_id + 1) * batch_size, :] = model100_l1.predict(vgg_blur_t)


                    t4 = time.time()
                    # print '\n finished prediction '
                    # print '\n time required for predicting outputs ' + str(t4 - t3)
                    del  vgg_dist_t , vgg_blur_t

                del vgg_ori_batch1, vgg_dist_batch , vgg_blur_batch

            vgg_ori.close()
            vgg_dist.close()
            vgg_blur.close()


            for filter_id in range(alexnet.layer_size101[layer_id, 0]):
                print " AWGN level : " + str(iter_id) + " , layer : " + str(alexnet.layer_names[layer_id]) + ", filter : " + str(filter_id)
                acc_dist[0,filter_id, iter_id],acc_dist[1,filter_id,iter_id] = base_acc.compute_test_accuracy(y_pred_dist[filter_id,:,:], yval)
                print " Blur level : " + str(iter_id) + " , layer : " + str(alexnet.layer_names[layer_id]) + ", filter : " + str(filter_id)
                acc_blur[0,filter_id, iter_id], acc_blur[1,filter_id,iter_id] = base_acc.compute_test_accuracy(y_pred_blur[filter_id,:,:], yval)

            del y_pred_dist , y_pred_blur

        out_file2.create_dataset("imagenet_awgn/" + str(alexnet.layer_names[layer_id]), data=acc_dist)
        out_file2.create_dataset("imagenet_blur/" + str(alexnet.layer_names[layer_id]), data=acc_blur)


        out_file2.close()




if filter_ranked == 0:

    # generate validation set for computing correction priorities
    IN_dat.imagenet_datagen()

    # compute baseline accuracy for distorted images in validation set
    base_acc.dc_ranksubsetacc()

    # generate layer ouptuts for computing correction priorities
    DNN_layers.get_alexnet_IN_layerout()

    #compute correction priorities
    get_layer_swap_acc()

    swap_acc =  h5py.File('alexnet_imagenet_layer_swap_acc.h5','r')

    # out_file = h5py.File('imagenet_alexnet_corr_acc_ideal.h5','w')

    dist_acc = h5py.File('imagenet_alexnet_ref_acc.h5','r')



    data_file = h5py.File('imagenet_dcranksubset_ori.h5', 'r')
    img_labl10 = data_file['img_label'][:]

    def sort_filters(dist_acc, corr_thresh=0.99):
        dist_chn = np.sort(dist_acc[:, 0])[::-1]
        dist_id = np.argsort(dist_acc[:, 0])[::-1]

        dist_pow = np.sum(np.square(dist_chn))
        dist_cd = np.zeros((dist_chn.shape[0], 1), np.float32)

        for filter_id in range(dist_chn.shape[0]):
            dist_cd[filter_id, 0] = dist_cd[max(filter_id - 1, 0), 0] + np.square(dist_chn[filter_id])

        dist_cd /= dist_pow
        print dist_id
        print '\n'
        print dist_cd.transpose()
        print '\n'
        print str(np.count_nonzero(dist_cd <= corr_thresh)) + " out of " + str(dist_chn.shape[0])
        return dist_id, dist_cd, np.count_nonzero(dist_cd <= corr_thresh)


    def get_corr_filter(layer_id):

        ref_acc10 = dist_acc['acc_mat'][:]

        ref_awgn_acc10 = 1. - ref_acc10[0, :, 1]
        ref_blur_acc10 = 1. - ref_acc10[0, :, 2]

        awgn_layer_acc10 = swap_acc['imagenet_awgn/' + str(alexnet.layer_names[layer_id])][:]
        awgn_layer_acc10 = awgn_layer_acc10[0, :, :]

        awgn_layer_acc10 = 1. - awgn_layer_acc10
        blur_layer_acc10 = swap_acc['imagenet_blur/' + str(alexnet.layer_names[layer_id])][:]
        print blur_layer_acc10.shape
        blur_layer_acc10 = blur_layer_acc10[0, :, :]
        blur_layer_acc10 = 1. - blur_layer_acc10

        blur_net_acc = np.zeros((alexnet.layer_size101[layer_id, 0], 1), np.float32)
        awgn_net_acc = np.zeros((alexnet.layer_size101[layer_id, 0], 1), np.float32)

        for dist_iter_id in range(IN_dat.num_dist):
            awgn_layer_acc10[:, dist_iter_id] = np.abs(
                awgn_layer_acc10[:, dist_iter_id] - ref_awgn_acc10[dist_iter_id]) / \
                                                ref_awgn_acc10[dist_iter_id]
            blur_layer_acc10[:, dist_iter_id] = np.abs(
                blur_layer_acc10[:, dist_iter_id] - ref_blur_acc10[dist_iter_id]) / \
                                                ref_blur_acc10[dist_iter_id]

        blur_net_acc10 = np.sum(blur_layer_acc10, axis=1, keepdims=True)
        awgn_net_acc10 = np.sum(awgn_layer_acc10, axis=1, keepdims=True)

        print '\n \n --------------------------- Layer ' + str(alexnet.layer_names[layer_id]) + '---------------\n'

        blur_id10, blur_cd10, blur_optcnt10 = sort_filters(blur_net_acc10)
        awgn_id10, awgn_cd10, awgn_optcnt10 = sort_filters(awgn_net_acc10)

        return awgn_id10, blur_id10

    ranked_fltr = h5py.File('imagenet_alexnet_ranked_filters.h5', 'w')
    for layer_id in range(0,len(alexnet.layer_names),1):

        awgn_id10, blur_id10 = get_corr_filter(layer_id)
        ranked_fltr.create_dataset('alexnet_awgn/layer_' + str(layer_id + 1), data=awgn_id10)
        ranked_fltr.create_dataset('alexnet_blur/layer_' + str(layer_id + 1), data=blur_id10)

    ranked_fltr.close()

    ranked_fltr = h5py.File('imagenet_alexnet_ranked_filters.h5','r')
    out_file = h5py.File('imagenet_alexnet_ideal_corr.h5','w')

    for layer_id in range(0,len(alexnet.layer_names),1):
        awgn_id10 = ranked_fltr['alexnet_awgn/layer_'+str(layer_id+1)][:]
        blur_id10 = ranked_fltr['alexnet_blur/layer_'+str(layer_id+1)][:]


        print "--------------------CIFAR 10 evaluation ----------------------"
        model10_l1, layer_dict10_l1 = alexnet.AlexNetDNN_layers('../Imagenet_models/alexnet_imagenet_weights.npy', layer_id=layer_id,
                                                               trainable=False, out_dim=out_dim)
        model10_l1.compile(optimizer=SGD(), loss='categorical_crossentropy')

        print '\n compiled model successfully '
        #
        acc_dist = np.empty((2,len(filt_steps), IN_dat.num_dist), np.float)
        acc_blur = np.empty((2,len(filt_steps), IN_dat.num_dist), np.float)

        filter_num = np.empty((len(filt_steps)),np.int32)
        for f_id in range(len(filt_steps)):
            filter_num[f_id] = filt_steps[f_id]*alexnet.layer_size101[layer_id,0]

        for iter_id in range(IN_dat.num_dist):

            vgg_ori = h5py.File('alexnet_imagenet_layer_outputs.h5','r')
            vgg_dist = h5py.File('alexnet_imagenet_layer_outputs_awgn_'+str(iter_id)+'.h5','r')
            vgg_blur = h5py.File('alexnet_imagenet_layer_outputs_blur_'+str(iter_id)+'.h5','r')


            y_pred_dist = np.empty((len(filter_num),num_test,out_dim), np.float)
            y_pred_blur = np.empty((len(filter_num),num_test,out_dim),np.float)

            for batch_id in range(0,num_test/batch_size):
                print '\n Processing batch : '+str(batch_id+1)
                t1 = time.time()

                vgg_ori_batch1 = vgg_ori[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()

                vgg_dist_batch = vgg_dist[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()

                vgg_blur_batch = vgg_blur[alexnet.layer_names[layer_id]][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()


                t2 = time.time()
                print '\n time required for loading file '+str(t2-t1)
                for corr_id in range(len(filter_num)):
                    vgg_dist_t = vgg_dist_batch.copy()
                    vgg_blur_t = vgg_blur_batch.copy()


                    for filter_id in range(filter_num[corr_id]):
                    #
                        vgg_dist_t[:,awgn_id10[filter_id],:,:] = vgg_ori_batch1[:,awgn_id10[filter_id],:,:].copy()

                    for filter_id in range(filter_num[corr_id]):

                        vgg_blur_t[:,blur_id10[filter_id],:,:] = vgg_ori_batch1[:,blur_id10[filter_id],:,:].copy()

                    y_pred_dist[corr_id,batch_id*batch_size:(batch_id+1)*batch_size,:] = model10_l1.predict(vgg_dist_t)


                    y_pred_blur[corr_id,batch_id*batch_size:(batch_id+1)*batch_size,:] = model10_l1.predict(vgg_blur_t)


                    del  vgg_dist_t , vgg_blur_t

                del vgg_ori_batch1, vgg_dist_batch , vgg_blur_batch

            vgg_ori.close()
            vgg_dist.close()
            vgg_blur.close()


            for grp_id in range(len(filter_num)):
                print " AWGN level : "+str(iter_id)+" , layer : "+str(alexnet.layer_names[layer_id])+", number of corrected filters : "+str(filter_num[grp_id])
                acc_dist[0,grp_id,iter_id],acc_dist[1,grp_id,iter_id] = base_acc.compute_test_accuracy(y_pred_dist[grp_id,:,:],img_labl10)
                print " Blur level : " + str(iter_id) + " , layer : " + str(alexnet.layer_names[layer_id]) + ", number of corrected filters : " + str(filter_num[grp_id])
                acc_blur[0,grp_id,iter_id], acc_blur[1,grp_id,iter_id] = base_acc.compute_test_accuracy(y_pred_blur[grp_id,:,:],img_labl10)


        out_file.create_dataset("awgn/"+str(alexnet.layer_names[layer_id]),data=acc_dist)
        out_file.create_dataset("blur/"+str(alexnet.layer_names[layer_id]),data=acc_blur)


    out_file.close()

ideal_corr_acc  = h5py.File('imagenet_alexnet_ideal_corr.h5','r')

# displat results
IN_blur_std = [1,2,3,4,5,6]


IN_awgn_std = [10,20,40,60,80,100]


colors = ['black','orange','red','green','blue']

fig = plt.figure(1)


for ly_id in range(5):
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.96, wspace=0.14, hspace=0.28)
    cifar_blur = ideal_corr_acc['blur/'+str(alexnet.layer_names[ly_id])][:]
    cifar_dist = ideal_corr_acc['awgn/'+str(alexnet.layer_names[ly_id])][:]
    ax = plt.subplot(4,3,ly_id+1)
    ax.plot(IN_blur_std,cifar_blur[0,0,:],'-s',markersize=8,markerfacecolor=str(colors[0]),markeredgecolor='k',color=str(colors[0]),label='top 10%' ,linewidth = 3)
    ax.plot(IN_blur_std,cifar_blur[0,1,:],'-o',markersize=8,markerfacecolor=str(colors[1]),markeredgecolor='k',color=str(colors[1]),label='top 25%',linewidth = 3)
    ax.plot(IN_blur_std,cifar_blur[0,2,:],'-v',markersize=8,markerfacecolor=str(colors[2]),markeredgecolor='k',color=str(colors[2]),label='top 50%',linewidth = 3)
    ax.plot(IN_blur_std,cifar_blur[0,3,:],'-p',markersize=8,markerfacecolor=str(colors[3]),markeredgecolor='k',color=str(colors[3]),label='top 75%',linewidth = 3)
    ax.plot(IN_blur_std,cifar_blur[0,4,:],'-h',markersize=8,markerfacecolor=str(colors[4]),markeredgecolor='k',color=str(colors[4]),label='top 90%',linewidth = 3)

    ax.set_title('Imagenet: Blur layer '+str(ly_id+1),fontsize=10)
    ax.title.set_position([0.5, 1.0])
    ax.set_xlabel('$\it{\sigma_b}$')
    plt.tick_params(axis='both', which='major')

    ax.set_ylabel('Top-1 accuracy')

    plt.grid(linestyle='dotted')

    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.96, wspace=0.14, hspace=0.28)
    ax = plt.subplot(4,3,7+ly_id)
    ax.plot(IN_awgn_std,cifar_dist[0,0,:],'-s',markersize=8,markerfacecolor=str(colors[0]),markeredgecolor='k',color=str(colors[0]),label='top 10%',linewidth = 3)
    ax.plot(IN_awgn_std,cifar_dist[0,1,:],'-o',markersize=8,markerfacecolor=str(colors[1]),markeredgecolor='k',color=str(colors[1]),label='top 25%',linewidth = 3)
    ax.plot(IN_awgn_std,cifar_dist[0,2,:],'-v',markersize=8,markerfacecolor=str(colors[2]),markeredgecolor='k',color=str(colors[2]),label='top 50%',linewidth = 3)
    ax.plot(IN_awgn_std,cifar_dist[0,3,:],'-p',markersize=8,markerfacecolor=str(colors[3]),markeredgecolor='k',color=str(colors[3]),label='top 75%',linewidth = 3)
    ax.plot(IN_awgn_std,cifar_dist[0,4,:],'-h',markersize=8,markerfacecolor=str(colors[4]),markeredgecolor='k',color=str(colors[4]),label='top 90%',linewidth = 3)

    ax.set_title('Imagenet: AWGN layer '+str(ly_id+1),fontsize=10)
    ax.title.set_position([0.5, 1.0])
    ax.set_xlabel('$\it{\sigma_n}$')
    ax.set_ylabel('Top-1 accuracy')
    plt.tick_params(axis='both', which='major')
    plt.grid(linestyle='dotted')



plt.legend(frameon=False,loc = 'upper center',bbox_to_anchor = (0.95,0.55), ncol=1,
        bbox_transform = plt.gcf().transFigure, prop={'size': 15})


fig.canvas.draw()
plt.show()



