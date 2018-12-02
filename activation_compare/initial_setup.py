import os
import argparse
import subprocess

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
            "--activations", help="the activations you use",
            required=False, default="relu",
            choices=["relu", "tanh", "leakyrelu", "prelu", "elu", "thresholdedrelu"]
            )
    p.add_argument(
            "--layerselect", help="whether or not change all activations",
            required=False, default="all", choices=["all", "input", "output"]
            )
    p_args = p.parse_args()
    activations = p_args.activations
    layer_select = p_args.layerselect

    return activations, layer_select

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        if os.name == "posix":
            command = "mkdir -p {}".format(dir_path)
            return_code = subprocess.call(command.split())
            assert return_code == 0,\
                    "Not create [{}]".format(dir_path)
        elif os.name == "nt":
            os.makedirs(dir_path)
        else:
            raise Exception("Unknown OS error...")

    return dir_path
