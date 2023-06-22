import tokenizers
import pickle
import argparse

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def learn_tokenizer_for_archive(arch):
    """
    the vocabulary used in the toy environment is actually very restricted, so it would be 
    computationally wasteful to use pretrained tokens from larger models
    """

    data_gen=(x for x in arch)



if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='tokenization utils')
    _parser.add_argument('--input_archive', type=str,  help="path to input archive", default="")

    _args=_parser.parse_args()

    with open(_args.input_archive,"rb") as fl:
        _arch=pickle.load(fl)




