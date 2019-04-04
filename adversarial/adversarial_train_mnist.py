"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans.plot import image as cleverhans_image

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
MODEL_PATH = os.path.join('models', 'adv_train_mnist')
TARGETED = True
NOISE_OUTPUT = True


def mnist_tutorial_adv_train(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=VIZ_ENABLED,
                      nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                      source_samples=SOURCE_SAMPLES,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=MODEL_PATH,
                      targeted=TARGETED,
                      noise_output=NOISE_OUTPUT):
  """
  MNIST tutorial for Adversarial Training
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param viz_enabled: (boolean) activate plots of adversarial examples
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param nb_classes: number of output classes
  :param source_samples: number of test inputs to attack
  :param learning_rate: learning rate for training
  :param model_path: path to the model file
  :param targeted: should we run a targeted attack? or untargeted?
  :return: an AccuracyReport object
  """
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  nb_filters = 64

  # Define TF model graph
  model = ModelBasicCNN('model1', nb_classes, nb_filters)
  preds = model.get_logits(x)
  loss = CrossEntropy(model, smoothing=0.1)
  print("Defined TensorFlow model graph.")

  ###########################################################################
  # Training the model using TensorFlow
  ###########################################################################

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'filename': os.path.split(model_path)[-1]
  }

  rng = np.random.RandomState([2017, 8, 30])
  # check if we've trained before, and if we have, use that pre-trained model
  if os.path.exists(model_path + ".meta"):
    tf_model_load(sess, model_path)
  else:
    train(sess, loss, x_train, y_train, args=train_params, rng=rng)
    saver = tf.train.Saver()
    saver.save(sess, model_path)

  # Evaluate the accuracy of the MNIST model on legitimate test examples
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  report.clean_train_clean_eval = accuracy

  ###########################################################################
  # Craft adversarial examples using FGSM - BIM - MIM approach
  ###########################################################################
  nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
  print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
        ' adversarial examples')
  print("This could take some time ...")

  # Instantiate a CW attack object
  fgsm = FastGradientMethod(model, sess=sess)
  bim = BasicIterativeMethod(model, sess=sess)
  mim = MomentumIterativeMethod(model, sess=sess)

  if viz_enabled:
    assert source_samples == nb_classes
    idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0]
            for i in range(nb_classes)]
  if targeted:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, 1, img_rows, img_cols,
                    nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')

      adv_inputs = np.array(
          [[instance] * nb_classes for instance in x_test[idxs]],
          dtype=np.float32)
    else:
      adv_inputs = np.array(
          [[instance] * nb_classes for
           instance in x_test[:source_samples]], dtype=np.float32)

    one_hot = np.zeros((nb_classes, nb_classes))
    one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

    adv_inputs = adv_inputs.reshape(
        (source_samples * nb_classes, img_rows, img_cols, nchannels))
    adv_ys = np.array([one_hot] * source_samples,
                      dtype=np.float32).reshape((source_samples *
                                                 nb_classes, nb_classes))
    yname = "y_target"
  else:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')

      adv_inputs = x_test[idxs]
    else:
      adv_inputs = x_test[:source_samples]

    adv_ys = None
    yname = "y"

  
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}
  bim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 50,
                  'eps_iter': .01}
  mim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 50,
                  'eps_iter': .01}

  adv_fgsm = fgsm.generate_np(adv_inputs,
                       **fgsm_params)
  adv_bim = bim.generate_np(adv_inputs,
                       **bim_params)
  adv_mim = mim.generate_np(adv_inputs,
                       **mim_params)
  eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
  if targeted:
    adv_fgsm_accuracy = model_eval(
        sess, x, y, preds, adv_fgsm, adv_ys, args=eval_params)
    adv_bim_accuracy = model_eval(
        sess, x, y, preds, adv_bim, adv_ys, args=eval_params)
    adv_mim_accuracy = model_eval(
        sess, x, y, preds, adv_mim, adv_ys, args=eval_params)
    
  else:
    if viz_enabled:
      err_fgsm = model_eval(sess, x, y, preds, adv_fgsm, y_test[idxs], args=eval_params)
      err_bim = model_eval(sess, x, y, preds, adv_bim, y_test[idxs], args=eval_params)
      err_mim = model_eval(sess, x, y, preds, adv_mim, y_test[idxs], args=eval_params)
      adv_fgsm_accuracy = 1 - err_fgsm
      adv_bim_accuracy = 1 - err_bim
      adv_mim_accuracy = 1 - err_mim
    else:
      err_fgsm = model_eval(sess, x, y, preds, adv_fgsm, y_test[:source_samples],
                       args=eval_params)
      err_bim = model_eval(sess, x, y, preds, adv_bim, y_test[:source_samples],
                      args=eval_params)
      err_mim = model_eval(sess, x, y, preds, adv_mim, y_test[:source_samples],
                      args=eval_params)
                      
      adv_fgsm_accuracy = 1 - err_fgsm
      adv_bim_accuracy = 1 - err_bim
      adv_mim_accuracy = 1 - err_mim


  print('--------------------------------------')

  # Compute the number of adversarial examples that were successfully found
  print('Avg. rate of successful adv. (FGSM) examples {0:.4f}'.format(adv_fgsm_accuracy))
  report.clean_train_adv_fgsm_eval = 1. - adv_fgsm_accuracy
  print('Avg. rate of successful adv. (BIM) examples {0:.4f}'.format(adv_bim_accuracy))
  report.clean_train_adv_bim_eval = 1. - adv_bim_accuracy
  print('Avg. rate of successful adv. (MIM) examples {0:.4f}'.format(adv_mim_accuracy))
  report.clean_train_adv_mim_eval = 1. - adv_mim_accuracy

  # Compute the average distortion introduced by the algorithm
  percent_perturbed_fgsm = np.mean(np.sum((adv_fgsm - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of (FGSM) perturbations {0:.4f}'.format(percent_perturbed_fgsm))
  percent_perturbed_bim = np.mean(np.sum((adv_bim - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of (BIM) perturbations {0:.4f}'.format(percent_perturbed_bim))
  percent_perturbed_mim = np.mean(np.sum((adv_mim - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of (MIM) perturbations {0:.4f}'.format(percent_perturbed_mim))
  
  ###########################################################################
  # Adversarial Training
  ###########################################################################

  model2 = ModelBasicCNN('model2', nb_classes, nb_filters) 
  
  fgsm2 = FastGradientMethod(model, sess=sess)
  # bim2 = BasicIterativeMethod(model, sess=sess)
  # mim2 = MomentumIterativeMethod(model, sess=sess)

  def attack_fgsm(x):
    return fgsm2.generate(adv_inputs, **fgsm_params)
  # def attack_bim(x):
  #   return bim2.generate(adv_inputs, **bim_params)
  # def attack_mim(x):
  #   return mim2.generate(adv_inputs, **mim_params)

  preds2 = model2.get_logits(x)
  loss2_fgsm = CrossEntropy(model2, smoothing=0.1, attack=attack_fgsm)
  # loss2_bim = CrossEntropy(model2, smoothing=0.1, attack=attack_bim)
  # loss2_mim = CrossEntropy(model2, smoothing=0.1, attack=attack_mim)

  train(sess, loss2_fgsm, x_train, y_train, args=train_params, rng=rng)
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, preds2, x_test, y_test, args=eval_params)
  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on adversarial fgsm test examples: {0}'.format(accuracy))
  report.clean_train_clean_eval = accuracy
  print("Defined TensorFlow model graph.")

  adv_fgsm_accuracy = model_eval(
        sess, x, y, preds, adv_fgsm, adv_ys, args=eval_params)
  adv_bim_accuracy = model_eval(
        sess, x, y, preds, adv_bim, adv_ys, args=eval_params)
  adv_mim_accuracy = model_eval(
        sess, x, y, preds, adv_mim, adv_ys, args=eval_params)

  # Close TF session
  sess.close()

  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial_adv_train(viz_enabled=FLAGS.viz_enabled,
                    nb_epochs=FLAGS.nb_epochs,
                    batch_size=FLAGS.batch_size,
                    source_samples=FLAGS.source_samples,
                    learning_rate=FLAGS.learning_rate,
                    attack_iterations=FLAGS.attack_iterations,
                    model_path=FLAGS.model_path,
                    targeted=FLAGS.targeted)


if __name__ == '__main__':
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Number of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('model_path', MODEL_PATH,
                      'Path to save or load the model file')
  flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                       'Number of iterations to run attack; 1000 is good')
  flags.DEFINE_boolean('targeted', TARGETED,
                       'Run the tutorial in targeted mode?')

  tf.app.run()
