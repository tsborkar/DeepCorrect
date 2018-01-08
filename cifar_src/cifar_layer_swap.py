

import numpy as np

import h5py
import matplotlib.pyplot as plt
np.random.seed(1337)
import time

from keras.optimizers import SGD

from keras.utils.np_utils import to_categorical


import cifar_layer_arch as CIFAR
import cifardata_gen as gen_cifar
import cifar_distortions as cifar_base_acc
import cifar_layers as cifar_layer_outputs


img_width, img_height = 32, 32
nchn = 3
layer_names = ['conv1_1','conv1_2','max_pool1','conv2_1','conv2_2','max_pool2','conv3_1','conv3_2','conv3_3','output']

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


#filter_ranked: 0 or 1
# set to 0 for recomputing the correction priorities
# set to 1 for using precomputed correction priorities

filter_ranked = 1



def compute_test_accuracy(y_pred, y_test):
    #    print('Testing on CIFAR-10')


    top1_err_drop = 0

    for test_id in range(len(y_pred)):

        if (np.argmax(y_pred[test_id, :]) != np.argmax(y_test[test_id, :])):
            top1_err_drop += 1

    accuracy = 1 - float(top1_err_drop) / float(len(y_test))
    print '\n Accuracy is : ' + str(accuracy)
    return accuracy



def get_layer_swap_acc():
    blur_std =[1,2,3,4,5,6]
    awgn_std =[5,10,20,30,40,50]
    dist_iter = 6
    num_test = 1000
    batch_size =  1000


    data_file = h5py.File('CIFAR_10_100_val.h5','r')

    img_labl100 = data_file['img_label100'][:]

    img_labl100 = to_categorical(img_labl100, nb_classes=100)
    print "----------------CIFAR 100 ---------------------------"
    out_dim  =  100
    for layer_id in range(0,6,1):
        out_file2 = h5py.File('cifar10_100_dist_layer_swap_acc100.h5', 'a')

        model100_l1, layer_dict100_l1 = CIFAR.cifar10_100_layer('../cifar_models/CIFAR_100_fine_best_model.npy', layer_id + 1, out_dim)
        model100_l1.compile(optimizer=SGD(), loss='categorical_crossentropy')

        print '\n compiled model successfully '

        acc_dist = np.empty((layer_size100[layer_id,0], dist_iter), np.float)
        acc_blur = np.empty((layer_size100[layer_id,0], dist_iter), np.float)

        acc_awgnblur = np.empty((layer_size100[layer_id,0], dist_iter), np.float)


        for iter_id in range(dist_iter):

            vgg_ori = h5py.File('cifar_layer_outputs.h5', 'r')
            vgg_dist = h5py.File('cifar_layer_outputs_dist_' + str(iter_id) + '.h5', 'r')
            vgg_blur = h5py.File('cifar_layer_outputs_blur_' + str(iter_id) + '.h5', 'r')


            y_pred_dist = np.empty((layer_size100[layer_id, 0], num_test, out_dim), np.float)
            y_pred_blur = np.empty((layer_size100[layer_id, 0], num_test, out_dim), np.float)



            for batch_id in range(0, num_test / batch_size):
                print '\n Processing batch : ' + str(batch_id + 1)
                t1 = time.time()
                vgg_ori_batch1 = vgg_ori['CIFAR_100/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size,:]

                vgg_dist_batch = vgg_dist['CIFAR_100_dist/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size,:]

                vgg_blur_batch = vgg_blur['CIFAR_100_blur/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :]



                t2 = time.time()
                print '\n time required for loading file ' + str(t2 - t1)

                for filter_id in range(layer_size100[layer_id, 0]):

                    vgg_dist_t = vgg_dist_batch.copy()
                    vgg_blur_t = vgg_blur_batch.copy()


                    t3 = time.time()

                    vgg_dist_t[:, filter_id, :, :] = vgg_ori_batch1[:, filter_id, :, :].copy()

                    y_pred_dist[filter_id, batch_id * batch_size:(batch_id + 1) * batch_size, :] = model100_l1.predict(vgg_dist_t)
                    # del vgg_ori_batch2

                    vgg_blur_t[:, filter_id, :, :] = vgg_ori_batch1[:, filter_id, :, :].copy()

                    y_pred_blur[filter_id, batch_id * batch_size:(batch_id + 1) * batch_size, :] = model100_l1.predict(vgg_blur_t)


                    t4 = time.time()

                    del vgg_blur_t, vgg_dist_t

                del vgg_ori_batch1, vgg_dist_batch, vgg_blur_batch

            vgg_ori.close()
            vgg_dist.close()
            vgg_blur.close()

            for filter_id in range(layer_size100[layer_id, 0]):
                print " AWGN level : " + str(iter_id) + " , layer : " + str(layer_names[layer_id]) + ", filter : " + str(filter_id)
                acc_dist[filter_id, iter_id] = compute_test_accuracy(y_pred_dist[filter_id,:,:], img_labl100)
                print " Blur level : " + str(iter_id) + " , layer : " + str(layer_names[layer_id]) + ", filter : " + str(filter_id)
                acc_blur[filter_id, iter_id] = compute_test_accuracy(y_pred_blur[filter_id,:,:], img_labl100)


        out_file2.create_dataset("CIFAR_100_dist/" + str(layer_names[layer_id]), data=acc_dist)
        out_file2.create_dataset("CIFAR_100_blur/" + str(layer_names[layer_id]), data=acc_blur)

        out_file2.close()



if filter_ranked == 0:

    # generate validation set for computing correction priorities
    gen_cifar.gen_cifar_val_data()

    # compute baseline accuracy for distorted images in validation set
    cifar_base_acc.get_baseline_val_acc()

    # generate layer ouptuts for computing correction priorities
    cifar_layer_outputs.get_Layer_output()

    #compute correction priorities
    get_layer_swap_acc()

    distortion_acc = h5py.File('CIFAR_10_100_valacc.h5', 'r')
    swap_acc = h5py.File('cifar10_100_dist_layer_swap_acc100.h5','r')
    filter_depths = [96,96,96,192,192,192]
    total_filters = np.sum(filter_depths)

    def sort_filters_global(dist_acc,corr_thresh=0.9):
        dist_chn = np.sort(dist_acc)[::-1]
        dist_id = np.argsort(dist_acc)[::-1]
        l2_sum, l1_sum = 0, 0
        # l2_seq = np.empty(len(dist_chn),np.float32)
        l1_seq = np.empty(len(dist_chn), np.float32)
        for score_id in range(len(dist_chn)):
            # l2_sum += dist_chn[score_id] * dist_chn[sc
            l1_sum += dist_chn[score_id]
            # l2_seq[score_id] = np.sqrt(l2_sum)
            l1_seq[score_id] = l1_sum
            # l2_seq /= np.sqrt(l2_sum)
        l1_seq /= l1_sum

        # print l2_seq
        print np.sum(l1_seq <= corr_thresh)

        return dist_id, np.sum(l1_seq <= corr_thresh)



    def get_corr_filt(layer_id):
        layer_names = ['conv1_1', 'conv1_2', 'max_pool1', 'conv2_1', 'conv2_2', 'max_pool2', 'conv3_1', 'conv3_2',
                       'conv3_3', 'output']

        ref_acc10 = distortion_acc['cifar_100'][:]


        ref_awgn_acc10 = 1. - ref_acc10[:, 1]
        ref_blur_acc10 = 1. - ref_acc10[:, 2]


        awgn_layer_acc10 = swap_acc['CIFAR_100_dist/' + str(layer_names[layer_id])][:]

        awgn_layer_acc10 = 1. - awgn_layer_acc10
        blur_layer_acc10 = swap_acc['CIFAR_100_blur/' + str(layer_names[layer_id])][:]
        blur_layer_acc10 = 1. - blur_layer_acc10


        for dist_iter_id in range(6):
            awgn_layer_acc10[:, dist_iter_id] = np.abs(awgn_layer_acc10[:, dist_iter_id] - ref_awgn_acc10[dist_iter_id]) / ref_awgn_acc10[dist_iter_id]
            blur_layer_acc10[:, dist_iter_id] = np.abs(blur_layer_acc10[:, dist_iter_id] - ref_blur_acc10[dist_iter_id]) / ref_blur_acc10[dist_iter_id]

        blur_net_acc10 = np.mean(blur_layer_acc10, axis=1, keepdims=True)
        awgn_net_acc10 = np.mean(awgn_layer_acc10, axis=1, keepdims=True)
        print '\n \n --------------------------- Layer ' + str(layer_names[layer_id]) + '---------------\n'
        awgn_id10, awgn_cnt, awgn_optcnt10 = sort_filters(awgn_net_acc10)
        blur_id10, blur_cnt, blur_optcnt10 = sort_filters(blur_net_acc10)

        return awgn_id10, blur_id10, awgn_cnt, blur_cnt

    def get_corr_filt_global(num_layers):
        layer_names = ['conv1_1', 'conv1_2', 'max_pool1', 'conv2_1', 'conv2_2', 'max_pool2', 'conv3_1', 'conv3_2',
                       'conv3_3', 'output']
        distortion_acc = h5py.File('CIFAR_10_100_acc.h5','r')
        ref_acc10 = np.array(distortion_acc.get('cifar_100'))

        ref_awgn_acc10 = 1. - ref_acc10[:, 1]
        ref_blur_acc10 = 1. - ref_acc10[:, 2]

        blur_all_layers = np.zeros(np.sum(filter_depths), np.float32)
        awgn_all_layers = np.zeros(np.sum(filter_depths),np.float32)
        start_id, end_id  = 0, 0
        for layer_id in range(num_layers):
            awgn_layer_acc10 = np.array(swap_acc.get('CIFAR_100_dist/' + str(layer_names[layer_id])))

            awgn_layer_acc10 = 1. - awgn_layer_acc10
            blur_layer_acc10 = np.array(swap_acc.get('CIFAR_100_blur/' + str(layer_names[layer_id])))

            blur_layer_acc10 = 1. - blur_layer_acc10

            for dist_iter_id in range(6):
                awgn_layer_acc10[:, dist_iter_id] = np.abs(awgn_layer_acc10[:, dist_iter_id] - ref_awgn_acc10[dist_iter_id]) / ref_awgn_acc10[dist_iter_id]
                blur_layer_acc10[:, dist_iter_id] = np.abs(blur_layer_acc10[:, dist_iter_id] - ref_blur_acc10[dist_iter_id]) / ref_blur_acc10[dist_iter_id]


            blur_net_acc10 = np.sum(blur_layer_acc10, axis=1, keepdims=True)
            awgn_net_acc10 = np.sum(awgn_layer_acc10, axis=1, keepdims=True)
            end_id += filter_depths[layer_id]
            blur_all_layers[start_id:end_id] = np.transpose(blur_net_acc10)
            awgn_all_layers[start_id:end_id] = np.transpose(awgn_net_acc10)

            start_id = end_id


        awgn_id10, awgn_cd10= sort_filters_global(awgn_all_layers)

        blur_id10, blur_cd10 = sort_filters_global(blur_all_layers)

        return awgn_id10, blur_id10, awgn_cd10, blur_cd10

    def sort_filters(dist_acc,corr_thresh=0.9):
        dist_chn = np.sort(dist_acc[:, 0])[::-1]
        dist_id = np.argsort(dist_acc[:, 0])[::-1]
        l2_sum, l1_sum = 0, 0

        l1_seq = np.empty(len(dist_chn),np.float32)
        for score_id in range(len(dist_chn)):

            l1_sum += dist_chn[score_id]

            l1_seq[score_id] = l1_sum

        l1_seq /= l1_sum

        print np.sum(l1_seq<=corr_thresh)

        return dist_id,np.sum(l1_seq<=corr_thresh),[]

    ranked_fltr = h5py.File('cifar_ranked_filters.h5','w')
    for ly_id in range(6):
        awgn_id10, blur_id10, awgn_cnt, blur_cnt = get_corr_filt(ly_id)
        ranked_fltr.create_dataset('CIFAR_100_awgn/layer_'+str(ly_id+1),data=awgn_id10)
        ranked_fltr.create_dataset('CIFAR_100_blur/layer_'+str(ly_id+1),data=blur_id10)

    ranked_fltr.close()


    dist_iter = 6
    num_test = 1000
    out_dim  = 100
    batch_size =  1000
    filt_steps = [0.1, 0.25, 0.5, 0.75, 0.9]
    print len(filt_steps)


    ranked_fltr = h5py.File('cifar_ranked_filters.h5','r')
    data_file = h5py.File('CIFAR_10_100_val.h5', 'r')
    # img_data_test = data_file['img_data100'][:]
    img_labl_test = to_categorical(data_file['img_label100'][:],out_dim)

    out_file = h5py.File('cifar_ideal_corr.h5','w')


    for layer_id in range(0,6,1):

        awgn_id10 = ranked_fltr['CIFAR_100_awgn/layer_'+str(layer_id+1)][:]
        blur_id10 = ranked_fltr['CIFAR_100_blur/layer_'+str(layer_id+1)][:]
        model10_l1, layer_dict10_l1 = CIFAR.cifar10_100_layer('../cifar_models/CIFAR_100_fine_best_model.npy', layer_id+1, out_dim)
        model10_l1.compile(optimizer=SGD(), loss='categorical_crossentropy')

        print '\n compiled model successfully '

        acc_dist = np.empty((len(filt_steps), dist_iter), np.float)
        acc_blur = np.empty((len(filt_steps), dist_iter), np.float)


        filter_num = np.empty((len(filt_steps)),np.int32)
        for f_id in range(len(filt_steps)):
            filter_num[f_id] = filt_steps[f_id]*layer_size100[layer_id,0]
        print filter_num


        for iter_id in range(dist_iter):

            vgg_ori = h5py.File('cifar_layer_outputs.h5','r')
            vgg_dist = h5py.File('cifar_layer_outputs_dist_'+str(iter_id)+'.h5','r')
            vgg_blur = h5py.File('cifar_layer_outputs_blur_'+str(iter_id)+'.h5','r')


            y_pred_dist = np.empty((len(filter_num),num_test,out_dim), np.float)
            y_pred_blur = np.empty((len(filter_num),num_test,out_dim),np.float)

            for batch_id in range(0,num_test/batch_size):
                print '\n Processing batch : '+str(batch_id+1)
                t1 = time.time()

                vgg_ori_batch1 = vgg_ori['CIFAR_100/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()

                vgg_dist_batch = vgg_dist['CIFAR_100_dist/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()

                vgg_blur_batch = vgg_blur['CIFAR_100_blur/' + str(layer_names[layer_id])][batch_size * batch_id:(batch_id + 1) * batch_size, :, :, :].copy()


                t2 = time.time()
                print '\n time required for loading file '+str(t2-t1)
                for corr_id in range(len(filter_num)):
                    vgg_dist_t = vgg_dist_batch.copy()
                    vgg_blur_t = vgg_blur_batch.copy()


                    for filter_id in range(filter_num[corr_id]):

                        vgg_dist_t[:,awgn_id10[filter_id],:,:] = vgg_ori_batch1[:,awgn_id10[filter_id],:,:].copy()

                        vgg_blur_t[:,blur_id10[filter_id],:,:] = vgg_ori_batch1[:,blur_id10[filter_id],:,:].copy()
                    #



                    y_pred_dist[corr_id,batch_id*batch_size:(batch_id+1)*batch_size,:] = model10_l1.predict(vgg_dist_t)


                    y_pred_blur[corr_id,batch_id*batch_size:(batch_id+1)*batch_size,:] = model10_l1.predict(vgg_blur_t)


                    del vgg_blur_t, vgg_dist_t

                del vgg_ori_batch1, vgg_dist_batch, vgg_blur_batch

            vgg_ori.close()
            vgg_dist.close()
            vgg_blur.close()




            for grp_id in range(len(filter_num)):
                print " AWGN level : "+str(iter_id)+" , layer : "+str(layer_names[layer_id])+", number of corrected filters : "+str(filter_num[grp_id])
                acc_dist[grp_id,iter_id] = compute_test_accuracy(y_pred_dist[grp_id,:,:],img_labl_test)
                print " Blur level : " + str(iter_id) + " , layer : " + str(layer_names[layer_id]) + ", number of corrected filters : " + str(filter_num[grp_id])
                acc_blur[grp_id,iter_id] = compute_test_accuracy(y_pred_blur[grp_id,:,:],img_labl_test)


        out_file.create_dataset('dist/'+str(layer_names[layer_id]),data=acc_dist)
        out_file.create_dataset('blur/'+str(layer_names[layer_id]),data=acc_blur)


    out_file.close()

ideal_corr_acc  = h5py.File('cifar_ideal_corr.h5','r')

# displat results
cifar_blur_std = [0.5,1,1.5,2,2.5,3]


cifar_awgn_std = [5,10,15,20,25,30]


colors = ['black','orange','red','green','blue']

fig = plt.figure(1)





for ly_id in range(6):
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.96, wspace=0.14, hspace=0.28)
    cifar_blur = ideal_corr_acc['blur/'+str(layer_names[ly_id])][:]
    cifar_dist = ideal_corr_acc['dist/'+str(layer_names[ly_id])][:]
    ax = plt.subplot(4,3,ly_id+1)
    ax.plot(cifar_blur_std,cifar_blur[0,:],'-s',markersize=8,markerfacecolor=str(colors[0]),markeredgecolor='k',color=str(colors[0]),label='top 10%' ,linewidth = 3)
    ax.plot(cifar_blur_std,cifar_blur[1,:],'-o',markersize=8,markerfacecolor=str(colors[1]),markeredgecolor='k',color=str(colors[1]),label='top 25%',linewidth = 3)
    ax.plot(cifar_blur_std,cifar_blur[2,:],'-v',markersize=8,markerfacecolor=str(colors[2]),markeredgecolor='k',color=str(colors[2]),label='top 50%',linewidth = 3)
    ax.plot(cifar_blur_std,cifar_blur[3,:],'-p',markersize=8,markerfacecolor=str(colors[3]),markeredgecolor='k',color=str(colors[3]),label='top 75%',linewidth = 3)
    ax.plot(cifar_blur_std,cifar_blur[4,:],'-h',markersize=8,markerfacecolor=str(colors[4]),markeredgecolor='k',color=str(colors[4]),label='top 90%',linewidth = 3)

    ax.set_title('CIFAR-100: Blur layer '+str(ly_id+1),fontsize=10)
    ax.title.set_position([0.5, 1.0])
    ax.set_xlabel('$\it{\sigma_b}$')
    plt.tick_params(axis='both', which='major')

    ax.set_ylabel('Top-1 accuracy')

    plt.grid(linestyle='dotted')

    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.96, wspace=0.14, hspace=0.28)
    ax = plt.subplot(4,3,7+ly_id)
    ax.plot(cifar_awgn_std,cifar_dist[0,:],'-s',markersize=8,markerfacecolor=str(colors[0]),markeredgecolor='k',color=str(colors[0]),label='top 10%',linewidth = 3)
    ax.plot(cifar_awgn_std,cifar_dist[1,:],'-o',markersize=8,markerfacecolor=str(colors[1]),markeredgecolor='k',color=str(colors[1]),label='top 25%',linewidth = 3)
    ax.plot(cifar_awgn_std,cifar_dist[2,:],'-v',markersize=8,markerfacecolor=str(colors[2]),markeredgecolor='k',color=str(colors[2]),label='top 50%',linewidth = 3)
    ax.plot(cifar_awgn_std,cifar_dist[3,:],'-p',markersize=8,markerfacecolor=str(colors[3]),markeredgecolor='k',color=str(colors[3]),label='top 75%',linewidth = 3)
    ax.plot(cifar_awgn_std,cifar_dist[4,:],'-h',markersize=8,markerfacecolor=str(colors[4]),markeredgecolor='k',color=str(colors[4]),label='top 90%',linewidth = 3)

    ax.set_title('CIFAR-100: AWGN layer '+str(ly_id+1),fontsize=10)
    ax.title.set_position([0.5, 1.0])
    ax.set_xlabel('$\it{\sigma_n}$')
    ax.set_ylabel('Top-1 accuracy')
    plt.tick_params(axis='both', which='major')
    plt.grid(linestyle='dotted')



plt.legend(frameon=False,loc = 'upper center',bbox_to_anchor = (0.95,0.55), ncol=1,
        bbox_transform = plt.gcf().transFigure, prop={'size': 15})


fig.canvas.draw()
plt.show()




