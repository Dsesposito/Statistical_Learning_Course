import numpy as np

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Data import DataAndLabels
from Data import DataSubSets
from random import randint


class ClassifierTrainCallBack(keras.callbacks.Callback):

    def __init__(self, test_data):
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        model_test_loss, model_test_acc = self.model.evaluate(
            x=self.test_data.data,
            y=self.test_data.labels,
            batch_size=100,
            verbose=1
        )

        self.train_losses.append(logs['loss'])
        self.test_accuracies.append(model_test_acc)

        self.train_accuracies.append(logs['acc'])
        self.test_losses.append(model_test_loss)

    def get_results(self):
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies


class AutoEncoderTrainCallBack(keras.callbacks.Callback):

    def __init__(self, test_data, noisy_test_data):
        self.test_data = test_data
        self.noisy_test_data = noisy_test_data
        self.train_losses = []
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        image_size = self.test_data.data.shape[2]

        model_test_loss = self.model.evaluate(
            x=self.noisy_test_data.data.reshape(len(self.noisy_test_data.data), image_size * image_size),
            y=self.test_data.data.reshape(len(self.test_data.data), image_size * image_size),
            batch_size=100,
            verbose=1
        )

        print('-------------------------------')
        print(logs['loss'])
        print(model_test_loss)
        print('-------------------------------')

        self.train_losses.append(logs['loss'])
        self.test_losses.append(model_test_loss)

    def get_results(self):
        return self.train_losses, self.test_losses


def create_data_sets():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    fashion_mnist = keras.datasets.fashion_mnist

    (train_data_complete_set, train_labels_complete_set), (test_images, test_labels) = fashion_mnist.load_data()

    complete_train_set = DataAndLabels(train_data_complete_set, train_labels_complete_set)
    complete_test_set = DataAndLabels(test_images, test_labels)

    num_train_samples = 2000
    num_validation_samples = 200
    num_test_samples = 10000

    data_sets = DataSubSets.build_from_complete_set(
        complete_train_set, complete_test_set,
        num_train_samples, num_test_samples, num_validation_samples,
        class_names
    )

    return data_sets


def create_classifier_model(weights=None):
    image_size = 28
    num_classes = 10
    layers_size = [512, 256]

    if weights is None:
        first_hidden_layer = keras.layers.Dense(units=layers_size[0], activation=tf.nn.relu)
        second_hidden_layer = keras.layers.Dense(units=layers_size[1], activation=tf.nn.relu)
    else:
        first_hidden_layer = keras.layers.Dense(units=layers_size[0], activation=tf.nn.relu, weights=weights[0:2])
        second_hidden_layer = keras.layers.Dense(units=layers_size[1], activation=tf.nn.relu, weights=weights[2:4])

    new_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(image_size, image_size)),
        first_hidden_layer,
        second_hidden_layer,
        keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)
    ])

    new_model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'],
    )

    return new_model


def train_classifier_model(data_sets, model_to_train, batch_size=100):
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    train_call_back = ClassifierTrainCallBack(test_data=data_sets.test)

    model_to_train.fit(
        x=data_sets.train.data,
        y=data_sets.train.labels,
        validation_data=(data_sets.validation_one.data, data_sets.validation_one.labels),
        batch_size=batch_size,
        epochs=50,
        #callbacks=[early_stop_callback, train_call_back],
        callbacks=[train_call_back],
        verbose=1
    )

    results = train_call_back.get_results()

    return results


def plot_results(train_losses, test_losses, train_accuracies=None, test_accuracies=None, file_base_name='', decimals=2):
    if train_accuracies is not None and test_accuracies is not None:
        plt.plot(train_accuracies)
        ax = plt.gca()
        ax.annotate(
            s='( {} , {:.' + str(decimals) + 'f} )'.format(len(train_accuracies), train_accuracies[-1]),
            xy=(len(train_accuracies) - 1, train_accuracies[-1]),
            horizontalalignment='left',
            verticalalignment='bottom'
        )
        plt.plot(test_accuracies)
        ax = plt.gca()
        ax.annotate(
            s='( {} , {:.' + str(decimals) + 'f} )'.format(len(test_accuracies), test_accuracies[-1]),
            xy=(len(test_accuracies) - 1, test_accuracies[-1]),
            horizontalalignment='left',
            verticalalignment='bottom'
        )
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(file_base_name + 'accuracy.png')
        plt.close(plt.gcf())

    plt.plot(train_losses)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.5f} )'.format(len(train_losses), train_losses[-1]),
        xy=(len(train_losses) - 1, train_losses[-1]),
        horizontalalignment='left',
        verticalalignment='bottom'
    )
    # plt.plot(test_losses)
    # ax = plt.gca()
    # ax.annotate(
    #     s='( {} , {:.' + str(decimals) + 'f} )'.format(len(test_losses), test_losses[-1]),
    #     xy=(len(test_losses) - 1, test_losses[-1]),
    #     horizontalalignment='left',
    #     verticalalignment='bottom'
    # )
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(file_base_name + 'loss.png')
    plt.close(plt.gcf())


def create_auto_encoder_model():
    input_shape = (784,)
    encoder_layers_sizes = [512, 256]
    decoder_layers_sizes = [512, 784]

    encoder_input = keras.Input(shape=input_shape)
    encoder = keras.layers.Dense(encoder_layers_sizes[0], activation='relu')(encoder_input)
    encoder = keras.layers.Dense(encoder_layers_sizes[1], activation='relu')(encoder)
    decoder = keras.layers.Dense(decoder_layers_sizes[0], activation='relu')(encoder)
    decoder = keras.layers.Dense(decoder_layers_sizes[1], activation='sigmoid')(decoder)

    autoencoder_model = keras.Model(encoder_input, decoder)

    autoencoder_model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        loss=keras.losses.mean_squared_error,
        metrics=['loss']
    )

    return autoencoder_model


def train_autoencoder_model(data_sets, model):
    image_size = 28

    train_call_back = AutoEncoderTrainCallBack(test_data=data_sets.test, noisy_test_data=data_sets.noisy_test)

    model.fit(
        x=data_sets.noisy_train.data.reshape(len(data_sets.noisy_train.data), image_size * image_size),
        y=data_sets.train.data.reshape(len(data_sets.train.data), image_size * image_size),
        epochs=10,
        batch_size=100,
        callbacks=[train_call_back]
    )

    return train_call_back.get_results()


def predict_images(model, data_sets):
    image_size = data_sets.noisy_test.data.shape[2]

    prediction_result = model.predict(
        data_sets.noisy_test.data.reshape(len(data_sets.noisy_test.data), image_size * image_size)
    )

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(prediction_result[i].reshape(image_size, image_size), cmap=plt.cm.binary)
        plt.xlabel(data_sets.class_names[data_sets.noisy_test.labels[i]])

    plt.savefig('autoencoder_predictions.png')
    plt.close(plt.gcf())

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data_sets.noisy_test.data[i].reshape(image_size, image_size), cmap=plt.cm.binary)
        plt.xlabel(data_sets.class_names[data_sets.noisy_test.labels[i]])

    plt.savefig('autoencoder_predictions_originals.png')
    plt.close(plt.gcf())


def transfer_weights(autoencoder_model, classifier_model):
    auto_encoder_weights = autoencoder_model.get_layer(index=2).get_weights()

    classifier_model.get_layer(index=2).set_weights(auto_encoder_weights)


def plot_comparison(first_results, second_results, title, ylabel, legend, file_name):
    plt.plot(first_results)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.2f} )'.format(len(first_results), first_results[-1]),
        xy=(len(first_results) - 1, first_results[-1]),
        horizontalalignment='left',
        verticalalignment='bottom'
    )
    plt.plot(second_results)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.2f} )'.format(len(second_results), second_results[-1]),
        xy=(len(second_results) - 1, second_results[-1]),
        horizontalalignment='left',
        verticalalignment='bottom'
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(file_name)
    plt.close(plt.gcf())


def main():
    # Exercise one
    data_sets = create_data_sets()

    # model = create_classifier_model()
    #
    # # Exercise two
    # train_losses, train_accuracies, test_losses, test_accuracies = train_classifier_model(
    #     data_sets, model)
    #
    # plot_results(
    #     train_losses=train_losses, train_accuracies=train_accuracies,
    #     test_losses=test_losses, test_accuracies=test_accuracies,
    #     file_base_name='classifier_'
    # )

    # Exercise tree
    auto_encoder_model = create_auto_encoder_model()

    autoencoder_train_losses, autoencoder_test_losses = train_autoencoder_model(data_sets, auto_encoder_model)

    plot_results(train_losses=autoencoder_train_losses, test_losses=autoencoder_test_losses,
                 file_base_name='auto_encoder_')

    predict_images(auto_encoder_model, data_sets)

    # # Exercise four
    #
    # encoder_weights = auto_encoder_model.get_weights()[0:4]
    #
    # classifier_model_with_autoencoder_initializer = create_classifier_model(encoder_weights)
    #
    # train_losses_with_pre_training, train_accuracies_with_pre_training, test_losses_with_pre_training, \
    # test_accuracies_with_pre_training = train_classifier_model(
    #     data_sets=data_sets,
    #     model=classifier_model_with_autoencoder_initializer
    # )
    #
    # plot_results(
    #     train_losses=train_losses_with_pre_training, train_accuracies=train_accuracies_with_pre_training,
    #     test_losses=test_losses_with_pre_training, test_accuracies=test_accuracies_with_pre_training,
    #     file_base_name='classifier_with_autoencoder_init_'
    # )
    #
    # plot_comparison(first_results=test_losses, second_results=test_losses_with_pre_training,
    #                 title='Loss with and without encoder pre-training',
    #                 ylabel='Loss',
    #                 legend=['Without pre-training', 'With pre-training'],
    #                 file_name='loss_comparison_encoder_pre_training')
    #
    # plot_comparison(test_accuracies, test_accuracies_with_pre_training,
    #                 title='Accuracies with and without encoder pre-training',
    #                 ylabel='Accuracies',
    #                 legend=['Without pre-training', 'With pre-training'],
    #                 file_name='accuracies_comparison_encoder_pre_training')
    #
    # # Exercise five
    # batches = [25, 50, 100, 200, 400]
    # for batch in batches:
    #     model = create_classifier_model()
    #     train_losses, train_accuracies, test_losses, test_accuracies = \
    #         train_classifier_model(data_sets, model, batch)
    #
    #     plt.plot(train_losses)
    #     plt.plot(train_accuracies)
    #
    #     tf.keras.backend.clear_session()
    #     del model, train_losses, train_accuracies, test_losses, test_accuracies
    #
    # plt.savefig("ex5")
    # plt.close(plt.gcf())