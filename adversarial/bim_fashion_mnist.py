"""
This tutorial shows how to generate adversarial examples
using BIM attack in white-box setting.
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

from cleverhans.attacks import BasicIterativeMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans.plot import image as cleverhans_image

from tensorflow import keras
from keras.utils import np_utils

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
BIM_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
MODEL_PATH = os.path.join('models', 'bim_fashion_mnist')
TARGETED = True
NOISE_OUTPUT = True


def mnist_tutorial_bim(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=VIZ_ENABLED,
                      nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                      source_samples=SOURCE_SAMPLES,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=MODEL_PATH,
                      targeted=TARGETED,
                      noise_output = NOISE_OUTPUT):
  """
  MNIST tutorial for Basic Iterative Method's attack
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

   # Get Fashion MNIST test data
  fashion = keras.datasets.fashion_mnist
  (x_train, y_train), (x_test, y_test) = fashion.load_data()
  # cifar10 = CIFAR10(train_start=train_start, train_end=train_end,
  #               test_start=test_start, test_end=test_end)
  # x_train, y_train = cifar10.get_set('train')
  # x_test, y_test = cifar10.get_set('test')
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  y_train = np_utils.to_categorical(y_train, 10)
  y_test = np_utils.to_categorical(y_test, 10)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

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
  # Craft adversarial examples using Basic Iterative Method's approach
  ###########################################################################
  nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
  print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
        ' adversarial examples')
  print("This could take some time ...")

  # Instantiate a BIM attack object
  bim = BasicIterativeMethod(model, sess=sess)

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
  else:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')

      adv_inputs = x_test[idxs]
    else:
      adv_inputs = x_test[:source_samples]

    adv_ys = None

  bim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 50,
                  'eps_iter': .01}

  adv = bim.generate_np(adv_inputs,
                       **bim_params)

  eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
  if targeted:
    adv_accuracy = model_eval(
        sess, x, y, preds, adv, adv_ys, args=eval_params)
  else:
    if viz_enabled:
      err = model_eval(sess, x, y, preds, adv, y_test[idxs], args=eval_params)
      adv_accuracy = 1 - err
    else:
      err = model_eval(sess, x, y, preds, adv, y_test[:source_samples],
                       args=eval_params)
      adv_accuracy = 1 - err

  if viz_enabled:
    for i in range(nb_classes):
      if noise_output:
        image = adv[i * nb_classes] - adv_inputs[i * nb_classes]
      else:
        image = adv[i * nb_classes]
      grid_viz_data[i, 0] = image

  print('--------------------------------------')

  # Compute the number of adversarial examples that were successfully found
  print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
  report.clean_train_adv_eval = 1. - adv_accuracy

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

  # Close TF session
  sess.close()
  def save_visual(data, path):
    """
    Modified version of cleverhans.plot.pyplot
    """
    figure = plt.figure()
    # figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    for y in range(num_rows):
      for x in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
        plt.axis('off')

        if num_channels == 1:
          plt.imshow(data[x, y, :, :, 0], cmap='gray')
        else:
          plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.savefig(path)
    return figure

  # Finally, block & display a grid of all the adversarial examples
  if viz_enabled:
    if noise_output:
      image_name = "output/bim_fashion_mnist_noise.png"
    else:
      image_name = "output/bim_fashion_mnist.png"
    _ = save_visual(grid_viz_data, image_name)

  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial_bim(viz_enabled=FLAGS.viz_enabled,
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
