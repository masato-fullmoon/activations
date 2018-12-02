from activation_compare.arrange_data import ArrangeData
from activation_compare.initial_setup import make_dirs
from activation_compare.ml_model import CompareActivations
import os

def dirs_init():
    datasets_path = make_dirs("./log_dir")
    weights_path = make_dirs("./weights_dir")
    return datasets_path, weights_path

if __name__ == "__main__":
    d_path, w_path = dirs_init()

    dataset = ArrangeData(dataset_path=d_path)

    (img_predict, label_predict), (img_train, label_train),\
            (img_val, label_val) = dataset()

    # CompareActivations(
    #         img_predict=img_predict, label_predict=label_predict,
    #         img_train=img_train, label_train=label_train,
    #         img_val=img_val, label_val=label_val
    #         )
    CompareActivations(
            log_path=d_path, weights_path=w_path,
            img_predict=img_predict, label_predict=label_predict,
            img_train=img_train, label_train=label_train,
            img_val=img_val, label_val=label_val,
            num_gpus=2
            )
