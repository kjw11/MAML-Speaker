#!/usr/bin/env python

import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import ImageDataGenerator
from maml import MASF

FLAGS = flags.FLAGS

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
except IndexError:
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Dataset PACS
flags.DEFINE_integer('num_classes', 800, 'number of classes used in classification.')

## Training options
flags.DEFINE_integer('train_iterations', 100000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 64, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 0.001, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 0.001, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 0.001, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './log/05102020', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 100, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 2000, 'intervals to save model')
flags.DEFINE_integer('print_interval', 100, 'intervals to print out training info')
flags.DEFINE_integer('infer_interval', 2000, 'intervals to test the model')


def suffule_line(l1, l2, l3):
    """Shuffle 3 list with same shuffle order."""
    lines, l1_new, l2_new, l3_new = [], [], [], []
    for i in range(len(l1)):
        lines.append((l1[i], l2[i], l3[i]))
    random.shuffle(lines)
    for i in range(len(lines)):
        l1_new.append(lines[i][0])
        l2_new.append(lines[i][1])
        l3_new.append(lines[i][2])
    return np.array(l1_new), np.array(l2_new), np.array(l3_new)


def train(model, saver, sess, exp_string, pairs_dir, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    support_losses, query_losses, support_accuracies, query_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(pairs_dir)):
            # get support set and query set
            pair_name = os.path.basename(pairs_dir[i])
            s_set, q_set = pair_name.split('-')
            s_set_list = os.path.join(pairs_dir[i], s_set+'.txt')
            q_set_list = os.path.join(pairs_dir[i], q_set+'.txt')
            
            s_set_data = ImageDataGenerator(s_set_list, batch_size=FLAGS.meta_batch_size, \
                                            num_classes=FLAGS.num_classes, shuffle=False) 
            q_set_data = ImageDataGenerator(q_set_list, batch_size=FLAGS.meta_batch_size, \
                                            num_classes=FLAGS.num_classes, shuffle=False) 
            s_iterator = tf.data.Iterator.from_structure(s_set_data.data.output_types, \
                                                             s_set_data.data.output_shapes)
            q_iterator = tf.data.Iterator.from_structure(q_set_data.data.output_types, \
                                                             q_set_data.data.output_shapes)
            tr_data_list.append((s_set_data, q_set_data))
            train_iterator_list.append((s_iterator, q_iterator))
            train_next_list.append((s_iterator.get_next(), q_iterator.get_next()))

    # Ops for initializing different iterators
    training_init_op = []
    s_batches_per_epoch, s_batch_marker = [], []
    q_batches_per_epoch, q_batch_marker = [], []
    for i in range(len(pairs_dir)):
        s_init = train_iterator_list[i][0].make_initializer(tr_data_list[i][0].data)
        q_init = train_iterator_list[i][1].make_initializer(tr_data_list[i][1].data)
        training_init_op.append((s_init, q_init))

        s_batches_per_epoch.append(int(np.floor(tr_data_list[i][0].data_size/FLAGS.meta_batch_size)))
        q_batches_per_epoch.append(int(np.floor(tr_data_list[i][1].data_size/FLAGS.meta_batch_size)))
        s_batch_marker = s_batches_per_epoch[:]
        q_batch_marker = q_batches_per_epoch[:]

    # Training begins
    print("Start training.")
    best_test_acc = 0
    start_time = time.time()
    # Initialize training iterator when itr=0 or it using out
    for i in range(len(pairs_dir)):
       sess.run(training_init_op[i][0])
       sess.run(training_init_op[i][1])
         
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sample a pair
        sampled_index = random.randint(0, len(pairs_dir)-1)
        sampled_pair = train_next_list[sampled_index]

                
        s_batch_marker[sampled_index]  = s_batch_marker[sampled_index]-1
        q_batch_marker[sampled_index]  = q_batch_marker[sampled_index]-1
        if s_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][0])
            s_batch_marker[sampled_index] = s_batches_per_epoch[sampled_index]
        if q_batch_marker[sampled_index] <= 0:
            sess.run(training_init_op[sampled_index][1])
            q_batch_marker[sampled_index] = q_batches_per_epoch[sampled_index]
        
        # Get sampled data
        inputa, labela ,namea = sess.run(sampled_pair[0])
        inputb, labelb ,nameb = sess.run(sampled_pair[1])

        # shuffle query set
        inputb, labelb, nameb = suffule_line(inputb, labelb, nameb)

        feed_dict = {model.inputa: inputa, model.labela: labela, \
                     model.inputb: inputb, model.labelb: labelb, \
                     model.KEEP_PROB: 1.0}

        output_tensors = [model.task_train_op]
        output_tensors.extend([model.summ_op, model.lossa, model.meta_loss, model.accuracya, model.accuracyb])
        _, summ_writer, support_loss, query_loss, support_acc, query_acc = sess.run(output_tensors, feed_dict)

        support_losses.append(support_loss)
        query_losses.append(query_loss)
        support_accuracies.append(support_acc)
        query_accuracies.append(query_acc)

        if itr % FLAGS.print_interval == 0:
            end_time = time.time()
            print('---'*10 + '\n%s' % exp_string)
            print('time %.4f s' % (end_time-start_time))
            print('Iteration %d' % itr + ': S Loss ' + str(np.mean(support_losses)))
            print('Iteration %d' % itr + ': Q Loss ' + str(np.mean(query_losses)))
            print('Iteration %d' % itr + ': S Accuracy ' + str(np.mean(support_accuracies)))
            print('Iteration %d' % itr + ': Q Accuracy ' + str(np.mean(query_accuracies)))
            support_losses, query_losses, target_losses = [], [], []
            start_time = time.time()

        if itr == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
            os.system('./infer.sh 1 2>&1 | tee -a infer.log ')
        if (itr!=0) and itr % FLAGS.save_interval == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
        if (itr!=0) and itr % FLAGS.infer_interval == 0:
            assert FLAGS.infer_interval % FLAGS.save_interval == 0
            os.system('./infer.sh 1 2>&1 | tee -a infer.log ')


def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line
    filelist_root = './train/pairs-ism'
    pairs = os.listdir('./train/pairs-ism')

    exp_string = 'ism'
    # Constructing model
    model = MASF(FLAGS.num_classes)
    model.construct_model_train()
    
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=tf.global_variables())
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    
    resume_itr = 0
    model_file = None
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    pairs_dir = [os.path.join(filelist_root, pair) for pair in pairs]
    train(model, saver, sess, exp_string, pairs_dir, resume_itr)
if __name__ == "__main__":
    main()
