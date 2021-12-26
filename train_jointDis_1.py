from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import jointDis

# set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

resolution = 256
teacher = 'DVA'
max_steps = 6000
batch_size = 64
lr = 0.001
num_threads = 8
distill_ratio = 1.0

train_frame_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/trainSet/Frame/{}/'.format(resolution)
train_ground_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/trainSet/Ground/{}/'.format(resolution)
train_teacher_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/trainSet/Teacher/{}/{}/'.format(resolution, teacher)
train_temporal_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/trainSet/Teacher/Temporal/{}/'.format(resolution)
valid_frame_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/validSet/Frame/{}/'.format(resolution)
valid_ground_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/validSet/Ground/{}/'.format(resolution)
valid_teacher_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/validSet/Teacher/{}/{}'.format(resolution, teacher)
valid_temporal_dir = '/media/cvmedia/Data/fukui/DIS/dataSet/AVS1K/validSet/Teacher/Temporal/{}/'.format(resolution)

save_dir = 'models/jointDis/{}/{}/default/'.format(resolution, teacher)
log_dir = 'logs/jointDis/{}/{}/default/'.format(resolution, teacher)


def make_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def list_all_files(dirname):
    all_files = []
    for i, j, k in os.walk(dirname):
        for l in k:
            all_files.append(os.path.join(i, l))
    return all_files


def data_input():
    train_frame_list = list_all_files(train_frame_dir)
    train_ground_list = list_all_files(train_ground_dir)
    train_teacher_list = list_all_files(train_teacher_dir)
    train_temporal_list = list_all_files(train_temporal_dir)
    valid_frame_list = list_all_files(valid_frame_dir)
    valid_ground_list = list_all_files(valid_ground_dir)
    valid_teacher_list = list_all_files(valid_teacher_dir)
    valid_temporal_list = list_all_files(valid_temporal_dir)
    train_next_frame_list = []
    for current_frame in train_frame_list:
        next_frame = '%s%05d.png' % (current_frame[:-9], int(current_frame[-9:-4]) + 1)
        if os.path.exists(next_frame):
            train_next_frame_list.append(next_frame)
        else:
            train_next_frame_list.append(current_frame)
    valid_next_frame_list = []
    for current_frame in valid_frame_list:
        next_frame = '%s%05d.png' % (current_frame[:-9], int(current_frame[-9:-4]) + 1)
        if os.path.exists(next_frame):
            valid_next_frame_list.append(next_frame)
        else:
            valid_next_frame_list.append(current_frame)

    # convert string into tensors
    train_frame_tensor = tf.convert_to_tensor(train_frame_list, dtype=tf.string)
    train_next_frame_tensor = tf.convert_to_tensor(train_next_frame_list, dtype=tf.string)
    train_ground_tensor = tf.convert_to_tensor(train_ground_list, dtype=tf.string)
    train_teacher_tensor = tf.convert_to_tensor(train_teacher_list, dtype=tf.string)
    train_temporal_ternsor = tf.convert_to_tensor(train_temporal_list, dtype=tf.string)
    valid_frame_tensor = tf.convert_to_tensor(valid_frame_list, dtype=tf.string)
    valid_next_frame_tensor = tf.convert_to_tensor(valid_next_frame_list, dtype=tf.string)
    valid_ground_tensor = tf.convert_to_tensor(valid_ground_list, dtype=tf.string)
    valid_teacher_tensor = tf.convert_to_tensor(valid_teacher_list, dtype=tf.string)
    valid_temporal_ternsor = tf.convert_to_tensor(valid_temporal_list, dtype=tf.string)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
        [train_frame_tensor, train_next_frame_tensor, train_ground_tensor,
         train_teacher_tensor, train_temporal_ternsor],
        shuffle=False,
    )
    valid_input_queue = tf.train.slice_input_producer(
        [valid_frame_tensor, valid_next_frame_tensor, valid_ground_tensor,
         valid_teacher_tensor, valid_temporal_ternsor],
        shuffle=False,
    )

    # process path and string tensor into an image and a label
    train_frame = tf.image.decode_png(tf.read_file(train_input_queue[0]), channels=3)
    train_next_frame = tf.image.decode_png(tf.read_file(train_input_queue[1]), channels=3)
    train_ground = tf.image.decode_png(tf.read_file(train_input_queue[2]), channels=1)
    train_teacher = tf.image.decode_png(tf.read_file(train_input_queue[3]), channels=1)
    train_temporal = tf.image.decode_png(tf.read_file(train_input_queue[4]), channels=1)
    valid_frame = tf.image.decode_png(tf.read_file(valid_input_queue[0]), channels=3)
    valid_next_frame = tf.image.decode_png(tf.read_file(valid_input_queue[1]), channels=3)
    valid_ground = tf.image.decode_png(tf.read_file(valid_input_queue[2]), channels=1)
    valid_teacher = tf.image.decode_png(tf.read_file(valid_input_queue[3]), channels=1)
    valid_temporal = tf.image.decode_png(tf.read_file(valid_input_queue[4]), channels=1)

    train_frame = tf.cast(train_frame, tf.float32) / 127.5 - 1
    train_next_frame = tf.cast(train_next_frame, tf.float32) / 127.5 - 1
    train_ground = tf.cast(train_ground, tf.float32) / 255
    train_teacher = tf.cast(train_teacher, tf.float32) / 255
    train_temporal = tf.cast(train_temporal, tf.float32) / 255
    valid_frame = tf.cast(valid_frame, tf.float32) / 127.5 - 1
    valid_next_frame = tf.cast(valid_next_frame, tf.float32) / 127.5 - 1
    valid_ground = tf.cast(valid_ground, tf.float32) / 255
    valid_teacher = tf.cast(valid_teacher, tf.float32) / 255
    valid_temporal = tf.cast(valid_temporal, tf.float32) / 255

    train_frame.set_shape((resolution, resolution, 3))
    train_next_frame.set_shape((resolution, resolution, 3))
    train_ground.set_shape((resolution, resolution, 1))
    train_teacher.set_shape((resolution, resolution, 1))
    train_temporal.set_shape((resolution, resolution, 1))
    valid_frame.set_shape((resolution, resolution, 3))
    valid_next_frame.set_shape((resolution, resolution, 3))
    valid_ground.set_shape((resolution, resolution, 1))
    valid_teacher.set_shape((resolution, resolution, 1))
    valid_temporal.set_shape((resolution, resolution, 1))

    # collect batches of images before processing
    train_frame_batch, train_next_frame_batch, train_ground_batch, train_teacher_batch, train_temporal_batch = tf.train.shuffle_batch(
        [train_frame, train_next_frame, train_ground, train_teacher,
         train_temporal],
        batch_size=batch_size,
        capacity=2000,
        min_after_dequeue=1000,
        num_threads=num_threads,
    )
    valid_frame_batch, valid_next_frame_batch, valid_ground_batch, valid_teacher_batch, valid_temporal_batch = tf.train.shuffle_batch(
        [valid_frame, valid_next_frame, valid_ground, valid_teacher,
         valid_temporal],
        batch_size=batch_size,
        capacity=2000,
        min_after_dequeue=1000,
        num_threads=num_threads,
    )
    train_data_batch = (train_frame_batch, train_next_frame_batch,
                        train_ground_batch, train_teacher_batch,
                        train_temporal_batch)
    valid_data_batch = (valid_frame_batch, valid_next_frame_batch,
                        valid_ground_batch, valid_teacher_batch,
                        valid_temporal_batch)
    return train_data_batch, valid_data_batch


def main():
    # create dir
    make_dir(save_dir)
    make_dir(log_dir+'/train')
    make_dir(log_dir+'/valid')

    with tf.Session() as sess:
        # placeholder
        frame = tf.placeholder(tf.float32,  shape=[None, resolution, resolution, 3], name='frame_input')
        next_frame = tf.placeholder(tf.float32,  shape=[None, resolution,
                                                        resolution, 3],
                                    name='next_frame_input')
        ground = tf.placeholder(tf.float32, shape=[None, resolution, resolution, 1], name='ground_input')
        teacher = tf.placeholder(tf.float32, shape=[None, resolution, resolution, 1], name='teacher_input')
        temporal = tf.placeholder(tf.float32, shape=[None, resolution,
                                                     resolution, 1],
                                  name='temporal_inpput')
        parse_train = tf.placeholder(tf.bool, name='parse_train')

        # model
        with slim.arg_scope(jointDis.jointDis_arg_scope(is_training=parse_train)):
            [student_s, student_t] = jointDis.jointDis(frame, next_frame)

        # loss
        loss_distill_s = tf.losses.mean_squared_error(student_s, teacher)
        loss_ground_s = tf.losses.mean_squared_error(student_s, ground)
        loss_s = distill_ratio * loss_distill_s + loss_ground_s
        loss_distill_t = tf.losses.mean_squared_error(student_t, teacher)
        loss_ground_t = tf.losses.mean_squared_error(student_t, ground)
        loss_t = distill_ratio * loss_distill_t + loss_ground_t

        # optimized
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op_s = slim.learning.create_train_op(loss_s, optimizer, update_ops=update_ops)
        train_op_t = slim.learning.create_train_op(loss_t, optimizer, update_ops=update_ops)

        # summary
        tf.summary.scalar('loss_distill_s', loss_distill_s)
        tf.summary.scalar('loss_ground_s', loss_ground_s)
        tf.summary.scalar('total_loss_s', loss_s)
        tf.summary.scalar('loss_distill_t', loss_distill_t)
        tf.summary.scalar('loss_ground_t', loss_ground_t)
        tf.summary.scalar('total_loss_t', loss_t)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(log_dir + '/valid')

        train_data_batch, valid_data_batch = data_input()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        def feed_dict(train):
            if train:
                data_batch = sess.run(train_data_batch)
            else:
                data_batch = sess.run(valid_data_batch)
            return {frame: data_batch[0], next_frame: data_batch[1],
                    ground: data_batch[2], teacher: data_batch[3],
                    temporal: data_batch[4], parse_train: train}

        # run
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        for i in range(max_steps):
            sess.run([train_op_s, train_op_t], feed_dict=feed_dict(True))
            if i%100 == 0:
                summary_train, loss_train = sess.run([merged, [loss_distill_s,
                                                               loss_ground_s,
                                                               loss_s,
                                                               loss_distill_t,
                                                               loss_ground_t,
                                                               loss_t]],
                                                     feed_dict=feed_dict(True))
                train_writer.add_summary(summary_train, i)
                print('train: step %d, loss_distill_s %.4f, loss_ground_s %.4f, total_loss_s %.4f, loss_distill_t %.4f, loss_ground_t %.4f, total_loss_t %.4f,' %
                      (i, loss_train[0], loss_train[1], loss_train[2],
                       loss_train[3], loss_train[4], loss_train[5]))
            # save and eval
            if i%1000 == 999 or i == max_steps-1:
                # save
                saver.save(sess, save_dir, global_step=i)
                print('* * * Adding checkpoint for ' + str(i) + ' * * *')
                # eval
                summary_valid, loss_valid = sess.run([merged, [loss_distill_s,
                                                               loss_ground_s,
                                                               loss_s,
                                                               loss_distill_t,
                                                               loss_ground_t,
                                                               loss_t]],
                                                     feed_dict=feed_dict(False))
                valid_writer.add_summary(summary_valid, i)
                #print('[val]: step %d, loss_distill %.4f, loss_ground %.4f, total_loss %.4f' %
                #      (i, loss_valid[0], loss_valid[1], loss_valid[2]))
                print('valid: step %d, loss_distill_s %.4f, loss_ground_s %.4f, total_loss_s %.4f, loss_distill_t %.4f, loss_ground_t %.4f, total_loss_t %.4f,' %
                      (i, loss_valid[0], loss_valid[1], loss_valid[2],
                       loss_valid[3], loss_valid[4], loss_valid[5]))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
