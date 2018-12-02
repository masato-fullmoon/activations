from keras.utils import to_categorical
import numpy as np
import os

class ArrangeData(object):
    def __init__(self, dataset_path=None, test_size=0.2, n_splits=5, prior=True, split="holdout"):
        if dataset_path is None:
            self.dataset_path = "~/log_dir"
        else:
            self.dataset_path = dataset_path
        self.test_size = test_size
        self.n_splits = n_splits
        self.split = split

        if not os.path.exists(os.path.join(self.dataset_path, "datasets.npz")) or prior == True:
            from keras.datasets import mnist
            (self.img_learning, self.label_learning),\
                    (self.img_predict, self.label_predict) = mnist.load_data()

            self.num_label = len(set(list(self.label_learning)))
            self.img_predict = self.img_predict.reshape(
                    self.img_predict.shape[0], self.img_predict.shape[1], self.img_predict.shape[2], 1
                    )
            self.img_predict = self.img_predict.astype("float32")/255.
            self.label_predict = to_categorical(self.label_predict, self.num_label)

            self.img_train, self.img_val,\
                    self.label_train, self.label_val = self.__split_data(
                            split=self.split
                            )
            np.savez(
                    os.path.join(self.dataset_path, "datasets.npz"),
                    img_predict=self.img_predict,
                    label_predict=self.label_predict,
                    img_train=self.img_train,
                    label_train=self.label_train,
                    img_val=self.img_val,
                    label_val=self.label_val
                    )
        else:
            if prior == False:
                prior_dataset = np.load(os.path.join(self.dataset_path, "datasets.npz"))
                self.img_predict = prior_dataset["img_predict"]
                self.label_predict = prior_dataset["label_predict"]
                self.img_train = prior_dataset["img_train"]
                self.label_train = prior_dataset["label_train"]
                self.img_val = prior_dataset["img_val"]
                self.label_val = prior_dataset["label_val"]

                self.num_label = len(set(list(self.label_predict)))
                print(self.num_label)
                print(self.img_predict)
            else:
                raise ValueError("prior is bool type...")

    def __call__(self):
        return (self.img_predict, self.label_predict),\
                (self.img_train, self.label_train),\
                (self.img_val, self.label_val)

    def __split_data(self, split):
        img_learning = self.img_learning.reshape(
                self.img_learning.shape[0], self.img_learning.shape[1], self.img_learning.shape[2], 1
                ) # (60000, 28, 28, 1)
        img_learning = img_learning.astype("float32")/255.
        label_learning = to_categorical(self.label_learning, self.num_label)

        if split == "holdout":
            from sklearn.model_selection import train_test_split
            if type(self.test_size) != float:
                raise TypeError("test_size is float type...")

            img_train, img_val,\
                    label_train, label_val = train_test_split(
                            img_learning, label_learning,
                            test_size=self.test_size
                            )
        elif split == "kfold":
            from sklearn.model_selection import KFold
            if type(self.n_splits) != int:
                raise TypeError("n_splits is integer type...")

            kf = KFold(n_splits=self.n_splits, shuffle=True)
            for t, v in kf.split(img_learning):
                img_train, img_val = img_learning[t], img_learning[v]
                label_train, label_val = label_learning[t], label_learning[v]
        else:
            raise ValueError("split is holdout or kfold...")

        return img_train, img_val, label_train, label_val
