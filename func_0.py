#--coding:utf-8--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import scipy.misc
from scipy import ndimage
from nets import spatiotemporal

# set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gen_result(imgPath, dstPath, modelPath, batch_size, resolution):
    if not os.path.isdir(dstPath):
        os.makedirs(dstPath)

    folderList = os.listdir(imgPath)

    imgList1 = []
    imgList2 = []

    for nfolder in folderList:
        frameList = os.listdir(imgPath + nfolder + '/')
        l = len(frameList)
        for idx in range(0, l):
            if idx == l - 1:
                imgList1.append(imgPath + nfolder + '/' + frameList[idx])
                imgList2.append(imgPath + nfolder + '/' + frameList[idx])
            else:
                imgList1.append(imgPath + nfolder + '/' + frameList[idx])
                imgList2.append(imgPath + nfolder + '/' + frameList[idx + 1])

    # placeholder
    frame_1 = tf.placeholder(tf.float32,  shape=[None, resolution, resolution,
                                                 3], name='frame_1_input')
    frame_2 = tf.placeholder(tf.float32, shape=[None, resolution, resolution,
                                                3], name='frame_2_input')

    # model
    with slim.arg_scope(spatiotemporal.spatiotemporal_arg_scope(is_training=False)):
        output = spatiotemporal.spatiotemporal(frame_1, frame_2)

    with tf.Session() as sess:
        # restore
        saver = tf.train.Saver()
        saver.restore(sess, modelPath+'-2999')

        len_set = len(imgList1)

        for idx in range(0, len_set, batch_size):
            imL1 = []
            imL2 = []
            sL = []
            bidx = 0
            while bidx < batch_size and idx + bidx < len_set:
                img1 = cv2.imread(imgList1[idx +
                                           bidx]).astype(np.float32)[:,:,[2,1,0]] / 127.5 -1
                img2 = cv2.imread(imgList2[idx + 
                                           bidx]).astype(np.float32)[:,:,[2,1,0]] / 127.5 -1
                sL.append(img1.shape)
                img1 = cv2.resize(img1, (resolution, resolution))
                img2 = cv2.resize(img2, (resolution, resolution))
                imL1.append(img1)
                imL2.append(img2)
                bidx += 1
            frameList1 = np.array(imL1)
            frameList2 = np.array(imL2)
            #print(type(frameList1))
            #print sL[0]
            start_time = time.time()
            #net_output = forward(frameList1, frameList2)
            net_output = sess.run(output,
                                    feed_dict={frame_1:frameList1,
                                               frame_2:frameList2})
            duration = time.time() - start_time
            print(net_output.shape[0])
            #img.resize(32,32)
            #scipy.misc.imsave(dstPath+'img.png', img)
            print('duration: %.5f, speed: %.5f, model path: %s' % (duration,
                                                                    batch_size/duration,
                                                                   modelPath))
            for k in range(0, net_output.shape[0]):
                tmp = net_output[k, :, :, :]
                tmp = cv2.resize(tmp, (sL[k][1], sL[k][0]))
                #blured = ndimage.gaussian_filter(tmp, sigma=3)
                blured = tmp
                min_value = np.min(blured)
                max_value = np.max(blured)
                blured = (blured - min_value)/(max_value - min_value) * 255.0
                sal_map = blured.astype(np.uint8)
                img_path = imgList1[idx + k]
                (filepath, frame) = os.path.split(img_path)
                nameidx = frame.rfind('_')
                savepath = dstPath + frame[:nameidx] + '/'
                if not os.path.isdir(savepath):
                    os.makedirs(savepath)
                scipy.misc.imsave(savepath + frame, sal_map)
                #print(savepath + frame)
    tf.reset_default_graph() 
if __name__ == '__main__':
    resList = {'32','64','96','128','256'}
    batch_size = 1
    teacher = 'SSNet'

    imgPath = '/media/cvmedia/Data/fukui/DIS/AVS1K/testSet/Frame/'
    for resolution in resList:
        dstPath = 'results-new/spatiotemporal/{}/{}/default/'.format(resolution, teacher)
        modelPath = 'models/spatiotemporal/{}/{}/default/'.format(resolution, teacher)
        gen_result(imgPath, dstPath, modelPath, batch_size, int(resolution))
