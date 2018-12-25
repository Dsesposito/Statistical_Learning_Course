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
        self.validation_losses = []
        self.validation_accuracies = []
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
        self.validation_losses.append(logs['val_loss'])
        self.test_accuracies.append(model_test_acc)

        self.train_accuracies.append(logs['acc'])
        self.validation_accuracies.append(logs['val_acc'])
        self.test_losses.append(model_test_loss)

    def get_results(self):
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies, \
               self.validation_losses, self.validation_accuracies


class AutoEncoderTrainCallBack(keras.callbacks.Callback):

    def __init__(self, test_data, noisy_test_data):
        self.test_data = test_data
        self.noisy_test_data = noisy_test_data
        self.train_losses = []
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        image_size = self.test_data.data.shape[2]

        reshaped_noisy_test_data = self.noisy_test_data.data.reshape(len(self.noisy_test_data.data),
                                                                     image_size * image_size)

        prediction_result = self.model.predict(reshaped_noisy_test_data)

        model_test_loss = np.mean((reshaped_noisy_test_data - prediction_result) ** 2)

        self.train_losses.append(logs['loss'])
        self.test_losses.append(model_test_loss)

    def get_results(self):
        return self.train_losses, self.test_losses


def create_data_sets():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    # class_names = range(0, 10)

    mnist = tf.keras.datasets.fashion_mnist

    (train_data_complete_set, train_labels_complete_set), (test_images, test_labels) = mnist.load_data()

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
        first_hidden_layer = keras.layers.Dense(
            units=layers_size[0],
            activation=tf.nn.relu
        )
        second_hidden_layer = keras.layers.Dense(
            units=layers_size[1],
            activation=tf.nn.relu
        )
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
        metrics=['accuracy']
    )

    return new_model


def train_classifier_model(data_sets, model, batch_size=100, use_early_stop=True, validation_set_to_use='One'):
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    train_call_back = ClassifierTrainCallBack(test_data=data_sets.test)

    if use_early_stop:
        callbacks = [early_stop_callback, train_call_back]
    else:
        callbacks = [train_call_back]

    if validation_set_to_use is 'Two':
        validation_set = (data_sets.validation_one.data, data_sets.validation_one.labels)
    else:
        validation_set = (data_sets.validation_two.data, data_sets.validation_two.labels)

    model.fit(
        x=data_sets.train.data,
        y=data_sets.train.labels,
        validation_data=validation_set,
        batch_size=batch_size,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    results = train_call_back.get_results()

    return results


def plot_results(train_losses, test_losses, train_accuracies=None, test_accuracies=None, file_base_name=''):
    if train_accuracies is not None and test_accuracies is not None:
        plt.plot(train_accuracies)
        ax = plt.gca()
        ax.annotate(
            s='( {} , {:.3f} )'.format(len(train_accuracies), train_accuracies[-1]),
            xy=(len(train_accuracies) - 1, train_accuracies[-1]),
            horizontalalignment='left',
            verticalalignment='bottom',
            xytext=(-60, 0),
            textcoords='offset pixels'
        )
        plt.plot(test_accuracies)
        ax = plt.gca()
        ax.annotate(
            s='( {} , {:.3f} )'.format(len(test_accuracies), test_accuracies[-1]),
            xy=(len(test_accuracies) - 1, test_accuracies[-1]),
            horizontalalignment='left',
            verticalalignment='bottom',
            xytext=(-60, 0),
            textcoords='offset pixels'
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
        s='( {} , {:.3f} )'.format(len(train_losses), train_losses[-1]),
        xy=(len(train_losses) - 1, train_losses[-1]),
        horizontalalignment='left',
        verticalalignment='bottom',
        xytext=(-60, 0),
        textcoords='offset pixels'
    )
    plt.plot(test_losses)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.3f} )'.format(len(test_losses), test_losses[-1]),
        xy=(len(test_losses) - 1, test_losses[-1]),
        horizontalalignment='left',
        verticalalignment='bottom',
        xytext=(-60, 0),
        textcoords='offset pixels'
    )
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
        loss=keras.losses.mean_squared_error
    )

    return autoencoder_model


def train_autoencoder_model(data_sets, model):
    image_size = 28

    train_call_back = AutoEncoderTrainCallBack(test_data=data_sets.test, noisy_test_data=data_sets.noisy_test)

    model.fit(
        x=data_sets.noisy_train.data.reshape(len(data_sets.noisy_train.data), image_size * image_size),
        y=data_sets.train.data.reshape(len(data_sets.train.data), image_size * image_size),
        epochs=500,
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


def plot_comparison(first_results, second_results, title, ylabel, legend, legend_location, offset_direction, file_name):
    plt.plot(first_results)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.2f} )'.format(len(first_results), first_results[-1]),
        xy=(len(first_results) - 1, first_results[-1])
    )
    plt.plot(second_results)
    ax = plt.gca()
    ax.annotate(
        s='( {} , {:.2f} )'.format(len(second_results), second_results[-1]),
        xy=(len(second_results) - 1, second_results[-1])
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(legend, loc=legend_location)
    plt.savefig(file_name)
    plt.close(plt.gcf())


def main():
    # Exercise one
    data_sets = create_data_sets()

    model = create_classifier_model()

    # Exercise two
    train_losses, train_accuracies, test_losses, test_accuracies, validation_losses, validation_accuracies = \
        train_classifier_model(
            data_sets=data_sets,
            model=model,
            use_early_stop=False
        )

    plot_results(
        train_losses=train_losses, train_accuracies=train_accuracies,
        test_losses=test_losses, test_accuracies=test_accuracies,
        file_base_name='classifier_'
    )

    # Exercise tree
    auto_encoder_model = create_auto_encoder_model()

    autoencoder_train_losses, autoencoder_test_losses = train_autoencoder_model(data_sets, auto_encoder_model)

    plot_results(train_losses=autoencoder_train_losses, test_losses=autoencoder_test_losses,
                 file_base_name='auto_encoder_')

    predict_images(auto_encoder_model, data_sets)

    # Exercise four

    encoder_weights = auto_encoder_model.get_weights()[0:4]

    classifier_model_with_autoencoder_initializer = create_classifier_model(encoder_weights)

    train_losses_with_pre_training, train_accuracies_with_pre_training, test_losses_with_pre_training, \
        test_accuracies_with_pre_training, validation_losses_with_pre_training, \
        validation_accuracies_with_pre_training = \
        train_classifier_model(
            data_sets=data_sets,
            model=classifier_model_with_autoencoder_initializer,
            use_early_stop=False
        )

    plot_results(
        train_losses=train_losses_with_pre_training, train_accuracies=train_accuracies_with_pre_training,
        test_losses=test_losses_with_pre_training, test_accuracies=test_accuracies_with_pre_training,
        file_base_name='classifier_with_autoencoder_pre_training_'
    )

    plot_comparison(first_results=test_losses, second_results=test_losses_with_pre_training,
                    title='Loss with and without encoder pre-training',
                    ylabel='Loss',
                    legend=['Without pre-training', 'With pre-training'],
                    legend_location='upper right',
                    offset_direction=1,
                    file_name='loss_comparison_encoder_pre_training')

    plot_comparison(test_accuracies, test_accuracies_with_pre_training,
                    title='Accuracies with and without encoder pre-training',
                    ylabel='Accuracies',
                    legend=['Without pre-training', 'With pre-training'],
                    legend_location='lower right',
                    offset_direction=-1,
                    file_name='accuracies_comparison_encoder_pre_training')

    # Exercise five
    batches = [25, 50, 100, 200, 400, 800, 1600]

    fig_loss = plt.figure()
    fig_acc = plt.figure()

    ax_loss = fig_loss.add_subplot(1, 1, 1)
    ax_acc = fig_acc.add_subplot(1, 1, 1)

    accuracies_per_model = []
    losses_per_model = []

    test_losses_per_model = []
    test_accuracies_per_model = []
    train_losses_per_model = []
    train_accuracies_per_model = []

    for i, batch in enumerate(batches):
        model = create_classifier_model()
        train_losses, train_accuracies, test_losses, test_accuracies, validation_losses, validation_accuracies = \
            train_classifier_model(data_sets, model, batch)

        accuracies_per_model.append(validation_accuracies[-1])
        losses_per_model.append(validation_losses[-1])

        test_losses_per_model.append(test_losses)
        test_accuracies_per_model.append(test_accuracies)

        train_losses_per_model.append(train_losses)
        train_accuracies_per_model.append(train_accuracies)

        ax_loss.plot(validation_losses)
        ax_loss.annotate(
            s='( {} , {:.2f} )'.format(len(validation_losses), validation_losses[-1]),
            xy=(len(validation_losses) - 1, validation_losses[-1])
        )

        ax_acc.plot(validation_accuracies)
        ax_acc.annotate(
            s='( {} , {:.2f} )'.format(len(validation_accuracies), validation_accuracies[-1]),
            xy=(len(validation_accuracies) - 1, validation_accuracies[-1])
        )

    legends = list(
        map(
            lambda batch: 'Batch size: {}'.format(batch),
            batches
        )
    )

    ax_loss.set_title('Losses comparison')
    ax_loss.set_ylabel('loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.legend(legends, loc='upper left')

    ax_acc.set_title('Accuracies comparison')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_xlabel('epoch')
    ax_acc.legend(legends, loc='lower right')

    fig_loss.savefig('losses_comparison_differents_batch_sizes.png')
    fig_acc.savefig('accuracies_comparison_differents_batch_sizes.png')

    plt.close(fig=fig_loss)
    plt.close(fig=fig_acc)

    best_model_index = int(np.argmax(accuracies_per_model))

    test_losses = test_losses_per_model[best_model_index]
    test_accuracies = test_accuracies_per_model[best_model_index]

    train_losses = train_losses_per_model[best_model_index]
    train_accuracies = train_accuracies_per_model[best_model_index]

    plot_results(
        train_losses=train_losses, train_accuracies=train_accuracies,
        test_losses=test_losses, test_accuracies=test_accuracies,
        file_base_name='classifier_batch_size_selection_'
    )
