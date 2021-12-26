#--coding:utf-8--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import shutil
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
import time
import scipy.misc
from scipy import ndimage
from nets import spatiotemporal
from PIL import Image

# set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.disable_eager_execution()

def frame2video(im_dir, video_dir, fps=30):

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    im_list = os.listdir(im_dir)
    im_list.sort()
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size
    video_name = im_dir.split('/')[-2] + '.webm'
    video_dir_ = os.path.join(video_dir, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    videoWriter = cv2.VideoWriter(video_dir_, fourcc, fps, img_size)

    for i in im_list:
        im_name = os.path.join(im_dir + i)
        #frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        frame = cv2.imread(im_name, cv2.IMREAD_COLOR)
        videoWriter.write(frame)
    videoWriter.release()
    print(video_dir_ + '.... finish')


def video2frame(video_path, frames_save_path):

    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)
    vidcap = cv2.VideoCapture(video_path)
    video_name = frames_save_path.split('/')[-2]
    success, image = vidcap.read()
    count = 1
    cv2.imencode('.png', image)[1].tofile(frames_save_path + video_name + '_%05d.png' % count)
    while success:
        success, image = vidcap.read()
        count += 1
        try:
            cv2.imencode('.png', image)[1].tofile(frames_save_path + video_name + '_%05d.png' % count)
        except:
            print(video_path + '  ... finish')
    print(count - 1)


def overlap(src, dst):
    dst = np.tile(dst[:,:,None], 3)
    im = src * dst
    return im

def gen_result(imgPath, dstPath, modelPath, batch_size, resolution):
    if not os.path.isdir(dstPath):
        os.makedirs(dstPath)

    folderList = os.listdir(imgPath)

    imgList1 = []
    imgList2 = []

    frameList = os.listdir(imgPath + '/')
    frameList.sort()
    l = len(frameList)
    for idx in range(0, l):
        if idx == l - 1:
            imgList1.append(imgPath + '/' + frameList[idx])
            imgList2.append(imgPath + '/' + frameList[idx])
        else:
            imgList1.append(imgPath + '/' + frameList[idx])
            imgList2.append(imgPath + '/' + frameList[idx + 1])

    # placeholder
    frame_1 = tf.placeholder(tf.float32,  shape=[None, resolution, resolution, 3], name='frame_s_input')
    frame_2 = tf.placeholder(tf.float32, shape=[None, resolution, resolution, 3], name='frame_t_input')

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
                blured = ndimage.gaussian_filter(tmp, sigma=3)
                #blured = tmp
                min_value = np.min(blured)
                max_value = np.max(blured)
                '''
                blured = (blured - min_value)/(max_value - min_value) * 255.0
                sal_map = blured.astype(np.uint8)
                '''
                blured = (blured - min_value)/(max_value - min_value)
                img_path = imgList1[idx + k]
                im = cv2.imread(img_path).astype(np.float32) / 255
                #sal_map = overlap(im, blured)
                sal_map = overlap(im, blured) * 255.0
                sal_map = sal_map.astype(np.uint8)
                (filepath, frame) = os.path.split(img_path)
                '''
                nameidx = frame.rfind('_')
                savepath = dstPath + frame[:nameidx] + '/'
                '''
                savepath = dstPath
                if not os.path.isdir(savepath):
                    os.makedirs(savepath)
                #scipy.misc.imsave(savepath + frame, sal_map)
                cv2.imwrite(savepath + frame, sal_map)
                #print(savepath + frame)
    tf.reset_default_graph() 
if __name__ == '__main__':
    src = '../Slomo_Flask-fk/web_slomo/static/upload/'
    dst = '../Slomo_Flask-fk/web_slomo/static/video/'
    modelPath = 'models/spatiotemporal/64/DVA/default/'
    fileList = os.listdir(src)
    for f in fileList:
        if f.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            gen_result(src, dst, modelPath, 1, 64)
        if f.endswith(('avi', 'mp4')):
            curSrc = 'res/Src/' + os.path.splitext(f)[0] + '/'
            curDst = 'res/Dst/' + os.path.splitext(f)[0] + '/'
            if os.path.exists(curSrc):
                shutil.rmtree(curSrc)
            if os.path.exists(curDst):
                shutil.rmtree(curDst)
            video2frame(src + f, curSrc)
            gen_result(curSrc, curDst, modelPath, 1, 64)
            frame2video(curSrc, src)
            frame2video(curDst, dst)
            os.remove(src + f)
            shutil.rmtree(curSrc)
            shutil.rmtree(curDst)
