from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.advanced_activations import ThresholdedReLU
from keras.callbacks import TensorBoard, CSVLogger
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

class CompareActivations(object):
    def __init__(
            self, log_path, weights_path, img_predict, label_predict,
            img_train, label_train, img_val, label_val,
            activations, layerselect,
            num_gpus=1, load_model=None, default=True, opt="adam"
            ):
        self.log_path = log_path
        self.weights_path = weights_path

        self.img_predict, self.label_predict = img_predict, label_predict
        self.img_train, self.label_train = img_train, label_train
        self.img_val, self.label_val = img_val, label_val

        self.num_label = len(self.label_train[0])
        print(self.num_label)

        self.activations = activations
        self.layerselect = layerselect
        self.num_gpus = num_gpus
        self.load_model = load_model
        self.default = default
        self.opt = opt

    def __call__(self):
        if self.default == True:
            self.feature_model, self.gpu_model = self.define_model()
        elif self.default == False and self.load_model:
            self.feature_model, self.gpu_model = self.prior_load(self.load_model)

        self.optimize(log_path=self.weights_path, opt=self.opt)

    def define_model(self):
        with tf.device("/cpu:0"):
            inputs = Input(self.img_train.shape[1:])
            if self.layerselect == "all":
                x = Conv2D(16, (5,5), padding="same")(inputs)
                x = Activations(activation=self.activations)(x)
                x = Conv2D(32, (5,5), padding="same")(x)
                x = Activations(activation=self.activations)(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                x = Conv2D(64, (5,5), padding="same")(x)
                x = Activations(activation=self.activations)(x)
                x = Conv2D(64, (5,5), padding="same")(x)
                x = Activations(activation=self.activations)(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                feature_model = Model(inputs, x)
                x = Flatten()(x)
                x = Dense(512)(x)
                x = Activations(activation=self.activations)(x)
                x = Dropout(0.5)(x)
            elif self.layerselect == "inputs":
                x = Conv2D(16, (5,5), padding="same")(inputs)
                x = Activations(activation=self.activations)(x)
                x = Conv2D(32, (5,5), padding="same", activation="relu")(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
                x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                feature_model = Model(inputs, x)
                x = Flatten()(x)
                x = Dense(512, activation="relu")(x)
                x = Dropout(0.5)(x)
            elif self.layerselect == "outputs"
                x = Conv2D(16, (5,5), padding="same", activation="relu")(inputs)
                x = Conv2D(32, (5,5), padding="same", activation="relu")(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
                x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
                x = MaxPooling2D((2,2))(x)
                x = Dropout(0.2)(x)
                feature_model = Model(inputs, x)
                x = Flatten()(x)
                x = Dense(512)(x)
                x = Activations(activation=sielf.activations)(x)
                x = Dropout(0.5)(x)
            outputs = Dense(self.num_label, activation="softmax")(x)
            total_model = Model(inputs, outputs)

            if self.num_gpus == 1:
                gpu_model = total_model
            elif self.num_gpus > 1:
                gpu_model = multi_gpu_model(total_model, gpus=self.num_gpus)

        return feature_model, gpu_model

    def prior_load(self, load_model):
        from keras.models import load_model
        prior_model = load_model(load_model)
        return prior_model

    def show_summary(self, model="total"):
        if model == "total":
            self.gpu_model.summary()
        elif model == "feature":
            self.feature_model.summary()
        else:
            raise ValueError("model argument is total or feature only ...")

    def optimization(self, log_path, opt="adam", lr=5e-5, beta_1=0.001, beta_2=0.9, decay=1e-6, momentum=0.9):
        if opt == "adam":
            from keras.optimizers import Adam
            optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
        elif opt == "sgd":
            from keras.optimizers import SGD
            optimizer = SGD(lr=lr, decay=decay, momentum=momentum)
        elif opt == "rmsprop":
            from keras.optimizers import rmsprop
            optimizer = rmsprop(lr=lr, decay=decay)
        else:
            raise ValueError("No optimizers...")

        self.feature_model.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer
                )
        self.feature_model.save(
                os.path.join(log_path, "{0}_feature.h5".format(opt))
                )
        self.gpu_model.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=LearningNorm(self.num_label)
                )

    def evaluation(self, log_path, batch_size=256, steps=50, verbose=0):
        tf_cb = TensorBoard(
                log_dir=os.path.join(log_dir, "tflogs"),
                histogram_freq=1
                )
        csv_log = CSVLogger(
                os.path.join(log_path, "{0}_logs.csv".format(self.opt))
                )
        history = self.gpu_model.fit(
                self.img_train, self.label_train,
                batch_size=batch_size, epochs=steps,
                callbacks=[tf_cb, csv_log],
                validation_data=(self.img_val, self.label_val),
                verbose=verbose
                )
        key_list = [key for key in history.history.keys() if not "val_" in key]
        for key in key_list:
            self.save_norms(
                    history=history, norm=key,
                    log_path=log_path
                    )

        score = self.gpu_model.evaluate(
                self.img_val, self.label_val,
                batch_size=batch_size, verbose=verbose
                )
        with open(os.path.join(log_path, "{0}_results.txt".format(self.opt)), "a") as results:
            results.write("---------- final validation results ----------\n")
            results.write("final val loss : {0}\n".format(score[0]))
            results.write("final val acc  : {0}\n".format(score[1]))
            results.write("\n")

    def prediction(self, log_path, verbose=0, log_verbose=False):
        predict = self.gpu_model.predict(
                self.img_predict, verbose=verbose
                )
        total = self.img_predict.shape[0]
        counter = 0
        proba_list = []
        if log_verbose == True:

        with open(os.path.join(log_path, "{0}_results.txt".format(self.opt)), "a") as p_result:
            p_result.write("---------- prediction results ----------\n")
            p_result.write("\n")
            for i in range(total):
                if np.argmax(self.label_predict[i]) == np.argmax(predict[i]):
                    counter += 1
                p_result.write(
                        "true:{0}\tprediction:{1}\n".format(
                            np.argmax(self.label_predict[i]),
                            np.argmax(predict[i])
                            )
                        )
                proba_list.append(np.amax(predict[i]))
            p_result.write("\n")
            p_result.write("---------- label-matching ----------\n")
            p_result.write("{0} [%]".format(counter/total*100))
            self.proba_hist(
                    proba_list=proba_list, log_path=log_path
                    )

    def save_norms(self, history, norm, log_path):
        train_info = histoty.history[norm]
        val_info = history.history["val_{0}".format(norm)]
        img_name = "{0}_{1}.png".format(self.opt, norm)

        plt.rcParams["font.size"] = 36
        plt.plot(train_info, label="train_{0}".format(norm), color="red", lw=5)
        plt.plot(val_info, label="val_{0}".format(norm), color="green", lw=5)
        plt.xlabel("Learning Steps")
        plt.ylabel("{0}".format(norm))
        plt.title("Learning Curve for {0}".format(norm))
        plt.legend()
        plt.savefig(os.path.join(log_path, img_name))

    def proba_hist(self, proba_list, log_path):
        df = pd.Series(proba_list)
        img_name = "{0}_hist.png".format(self.opt)

        plt.rcParams["font.size"] = 36
        plt.hist(
                df, lw=5, color="green", ec="black",
                bins=100, normed=True
                )
        plt.xlabel("Normed Probability")
        plt.ylabel("Frequency")
        plt.title("Frequency on Normalized Max Class Probability")
        plt.savefig(os.path.join(log_path, img_name))

class Activations(object):
    def __init__(self, activation, alpha=0.2):
        self.activation = activation
        self.alpha = alpha

    def __call__(self, inputs):
        if self.activation == "relu":
            activations_layer = Activation("relu")(inputs)
        elif self.activation == "tanh":
            activations_layer = Activation("tanh")(inputs)
        elif self.activation == "leakyrelu":
            activations_layer = LeakyReLU(self.alpha)(inputs)
        elif self.activation == "prelu":
            activations_layer = PReLU(self.alpha)(inputs)
        elif self.activation == "elu":
            activations_layer = ELU(self.alpha)(inputs)
        elif self.activation == "thresholdedrelu":
            activations_layer = ThresholdedReLU(self.alpha)(inputs)

        return activations_layer

class LearningNorm(object):
    def __init__(self, num_label):
        total_metrics = self.generate_metrics(num_label)
        return total_metrics

    def normalize_y_pred(self, y_pred):
        return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])

    def class_true_positive(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2),
                      K.floatx())

    def class_accuracy(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true[:, class_label], y_pred[:, class_label]),
                      K.floatx())

    def class_precision(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.sum(self.class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_pred[:, class_label]) + K.epsilon())

    def class_recall(self, class_label, y_true, y_pred):
        return K.sum(self.class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_true[:, class_label]) + K.epsilon())

    def class_f_measure(self, class_label, y_true, y_pred):
        precision = self.class_precision(class_label, y_true, y_pred)
        recall = self.class_recall(class_label, y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def true_positive(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true + y_pred, 2),
                      K.floatx())

    def micro_precision(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.sum(self.true_positive(y_true, y_pred)) / (K.sum(y_pred) + K.epsilon())

    def micro_recall(self, y_true, y_pred):
        return K.sum(self.true_positive(y_true, y_pred)) / (K.sum(y_true) + K.epsilon())

    def micro_f_measure(self, y_true, y_pred):
        precision = self.micro_precision(y_true, y_pred)
        recall = self.micro_recall(y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def average_accuracy(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        class_acc_list = [self.class_accuracy(i, y_true, y_pred) for i in range(class_count)]
        class_acc_matrix = K.concatenate(class_acc_list, axis=0)
        return K.mean(class_acc_matrix, axis=0)

    def macro_precision(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        return K.sum([self.class_precision(i, y_true, y_pred) for i in range(class_count)]) \
               / K.cast(class_count, K.floatx())

    def macro_recall(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        return K.sum([self.class_recall(i, y_true, y_pred) for i in range(class_count)]) \
               / K.cast(class_count, K.floatx())

    def macro_f_measure(self, y_true, y_pred):
        precision = self.macro_precision(y_true, y_pred)
        recall = self.macro_recall(y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def generate_metrics(self, num_label):
        metrics = ["accuracy"]

        # the metrics a class label
        func_list = [self.class_accuracy, self.class_precision, self.class_recall, self.class_f_measure]
        name_list = ["acc", "precision", "recall", "f_measure"]
        for i in range(num_label):
            for func, name in zip(func_list, name_list):
                func = partial(func, i)
                func.__name__ = "{0}-{1}".format(name, i)
                metrics.append(func)

        # total metrics
        metrics.append(self.average_accuracy)
        metrics.append(self.macro_precision)
        metrics.append(self.macro_recall)
        metrics.append(self.macro_f_measure)

        return metrics
