#!/usr/bin/env python

import re
import sys
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from maml import MASF
from kaldi_io import write_vec_flt, open_or_fd, read_mat_ark, read_mat_scp, read_vec_flt_ark
FLAGS = flags.FLAGS
from tensorflow.python import pywrap_tensorflow

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
except IndexError:
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Infer option
flags.DEFINE_string('model_path', './log/04082020/masf_singing.mbs_64.inner0.001.outer0.001.clipNorm2.0.metric0.001.margin10.0', 'model path')
flags.DEFINE_string('ark_file', './CNdata/train/feats.ark', 'infered vector path')
flags.DEFINE_string('out_file', './output/800-singing/', 'output.ark dir')

def predict_ark(model, ark_file_path, out_file_path, sess, vec_nom=False):

    # Testing periodically
    out_file_path = out_file_path+'/output.ark'
    out_file = open_or_fd(out_file_path, 'wb')
    for index, (key, feature) in enumerate(read_mat_ark(ark_file_path)):
        test_input = feature
        #feed_dict = {model.test_input: test_input, model.test_label: test_label, model.KEEP_PROB: 1.}
        feed_dict = {model.test_input: test_input}
        output_tensors = [model.embeddings]
        embeddings = sess.run(output_tensors, feed_dict)
        if vec_nom:
            embeddings /=np.sqrt(np.num(np.square(embeddings)))
        write_vec_flt(out_file ,embeddings[0][0], key=key)
        #print type(embeddings[0][0])
        #write_txt(out_file_path ,embeddings[0][0], key)

def predict_scp(model, scp_file_path, out_file_path, sess, vec_nom=False):

    # Testing periodically
    out_file_path = out_file_path+'/test/enroll.ark'
    out_file = open_or_fd(out_file_path, 'wb')
    lenth = 110000
    with tqdm(total=lenth) as pbar:
        for index, (key, feature) in enumerate(read_mat_scp(scp_file_path)):
            pbar.update(1)
            test_input = feature
            feed_dict = {model.test_input: test_input}
            output_tensors = [model.embeddings]
            embeddings = sess.run(output_tensors, feed_dict)
            if vec_nom:
                embeddings /=np.sqrt(np.num(np.square(embeddings)))
            write_vec_flt(out_file ,embeddings[0][0], key=key)


def write_txt(out_file, ndarray, key):
    with open(out_file, 'w') as f:
        f.write(key+'  ')
        f.write(str(ndarray))
        f.write('\n')

def load(path, sess):
    """Load the saved variables.

    If the variables have values, the current values will be changed to the saved ones
    :return The step of the saved model.
    """
    ckpt = tf.train.get_checkpoint_state(path)
    print path	
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print (ckpt_name)
        step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(path, ckpt_name))
        print("Succeed to load checkpoint {}".format(ckpt_name))
    else:
        sys.exit("Failed to find a checkpoint in {}".format(path))
    return step


def main():

    model_path = FLAGS.model_path 
    ark_file = FLAGS.ark_file
    out_file = FLAGS.out_file

    tf.reset_default_graph()

    if not os.path.exists(out_file):
        os.makedirs(out_file)

    # Constructing model
    model = MASF(FLAGS.num_classes)
    model.construct_model_predict(prefix='predict')

    sess = tf.Session() 
    init = tf.global_variables_initializer()
    sess.run(init)

    load(model_path, sess)
    predict_ark(model, ark_file, out_file, sess)


if __name__ == "__main__":
    main()
