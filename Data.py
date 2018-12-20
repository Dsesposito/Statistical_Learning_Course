import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf


class DataAndLabels:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class DataSubSets:
    def __init__(self, train, validation_one, validation_two, test, class_names, noisy_train=None, noisy_test=None,
                 noisy_validation_one=None):
        self.train = train
        self.validation_one = validation_one
        self.validation_two = validation_two
        self.test = test
        self.class_names = class_names
        self.noisy_train = noisy_train
        self.noisy_test = noisy_test
        self.noisy_validation_one = noisy_validation_one

    @staticmethod
    def build_from_complete_set(complete_train_set, complete_test_set, num_train_samples, num_test_samples,
                                num_validation_samples, class_names):
        randoms = np.random.permutation(num_train_samples + 2 * num_validation_samples)

        train_data_set, validation_data_set_one, validation_data_set_two = \
            DataSubSets.__split_set(complete_train_set.data, randoms, num_train_samples, num_validation_samples)

        train_labels_set, validation_labels_set_one, validation_labels_set_two = \
            DataSubSets.__split_set(complete_train_set.labels, randoms, num_train_samples, num_validation_samples)

        train_data_labels = DataAndLabels(train_data_set, train_labels_set)
        validation_one_data_labels = DataAndLabels(validation_data_set_one, validation_labels_set_one)
        validation_two_data_labels = DataAndLabels(validation_data_set_two, validation_labels_set_two)

        test_randoms = np.random.permutation(num_test_samples)
        test_data_set = complete_test_set.data[test_randoms]
        test_labels_set = complete_test_set.labels[test_randoms]

        test_data_labels = DataAndLabels(test_data_set, test_labels_set)

        image_size = train_data_labels.data.shape[2]
        rotation_from_angle = -45
        rotation_to_angle = 45
        im_ph = tf.placeholder(dtype=tf.int32, shape=(image_size, image_size))
        ang_ph = tf.placeholder(dtype=tf.float32, shape=())
        rot_op = tf.contrib.image.rotate(im_ph, ang_ph)

        noisy_train_data = np.zeros(shape=(len(train_data_labels.data), image_size, image_size))
        noisy_test_data = np.zeros(shape=(len(test_data_labels.data), image_size, image_size))
        noisy_validation_one_data = np.zeros(shape=(len(validation_one_data_labels.data), image_size, image_size))
        with tf.Session() as sess:
            for index, curr_image in enumerate(train_data_labels.data):
                rotation_angle = random.randint(rotation_from_angle, rotation_to_angle) * math.pi * 2 / 360
                rotated_image = sess.run(
                    fetches=rot_op,
                    feed_dict={im_ph: curr_image, ang_ph: np.float(rotation_angle)}
                )
                noisy_train_data[index] = rotated_image

            for index, curr_image in enumerate(test_data_labels.data):
                rotation_angle = random.randint(rotation_from_angle, rotation_to_angle) * math.pi * 2 / 360
                rotated_image = sess.run(
                    fetches=rot_op,
                    feed_dict={im_ph: curr_image, ang_ph: np.float(rotation_angle)}
                )
                noisy_test_data[index] = rotated_image

            for index, curr_image in enumerate(validation_one_data_labels.data):
                rotation_angle = random.randint(rotation_from_angle, rotation_to_angle) * math.pi * 2 / 360
                rotated_image = sess.run(
                    fetches=rot_op,
                    feed_dict={im_ph: curr_image, ang_ph: np.float(rotation_angle)}
                )
                noisy_validation_one_data[index] = rotated_image

        noisy_train_data_labels = DataAndLabels(
            data=noisy_train_data,
            labels=train_data_labels.labels
        )

        noisy_test_data_lables = DataAndLabels(
            data=noisy_test_data,
            labels=test_data_labels.labels
        )

        noisy_validation_one_data_labels = DataAndLabels(
            data=noisy_validation_one_data,
            labels=validation_one_data_labels.labels
        )

        train_data_labels.data = train_data_labels.data / 255
        validation_one_data_labels.data = validation_one_data_labels.data / 255
        validation_two_data_labels.data = validation_two_data_labels.data / 255
        noisy_train_data_labels.data = noisy_train_data_labels.data / 255
        noisy_test_data_lables.data = noisy_test_data_lables.data / 255
        noisy_validation_one_data_labels.data = noisy_validation_one_data_labels.data / 255

        return DataSubSets(
            train=train_data_labels,
            validation_one=validation_one_data_labels,
            validation_two=validation_two_data_labels,
            test=test_data_labels,
            noisy_train=noisy_train_data_labels,
            noisy_test=noisy_test_data_lables,
            noisy_validation_one=noisy_validation_one_data_labels,
            class_names=class_names
        )

    @staticmethod
    def __split_set(complete_set, randoms, num_test_samples, num_validation_samples):
        train_set = complete_set[
            randoms[
                0: num_test_samples
            ]
        ]

        validation_set_one = complete_set[
            randoms[
                num_test_samples: num_test_samples + num_validation_samples
            ]
        ]

        validation_set_two = complete_set[
            randoms[
                num_test_samples + num_validation_samples: num_test_samples + 2 * num_validation_samples
            ]
        ]

        return train_set, validation_set_one, validation_set_two
