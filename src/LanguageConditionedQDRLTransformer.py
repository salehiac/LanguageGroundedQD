import nanoGPT_QDRL
import yaml
import argparse


def main_train(cfg):
    pass

def main_train(cfg):
    pass

if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='Train/Test the mdoel.')
    _parser.add_argument('--config', type=str,  help="yaml config", default="")

    _args=_parser.parse_args()

    with open(_args.config,"r") as fl:
        _config=yaml.load(fl,Loader=yaml.FullLoader)

    if _config["train_model"]:
        _out_train=main_train(_config)
    if _config["test_model"]:
        _out_test=main_test(_config)



