import tokenizers
import pickle
import argparse
import pdb
from termcolor import colored
import matplotlib.pyplot as plt
import functools
import numpy as np
import string
import torch

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

def learn_tokenizer_for_archive(arch, min_frequ=3,save_to=""):
    """
    returns wrapped_tokenizer, tokenizer. In general, you want the first, the second is just usuful for debug
    """

   
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()#Whitespace chains Whitespacesplit() and punctuation() sequentially
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=5000, special_tokens=special_tokens,min_frequency=min_frequ, show_progress=True,boulip_moulip=True)

    tokenizer.train_from_iterator([x._llm_descr for x in arch],trainer=trainer)
    print(colored(f"trained tokenizer final vocab_size=={tokenizer.get_vocab_size()}","green",attrs=["bold"]))

    tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",#we don't care about pairs
            special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
            )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        return_tensors="pt"
        )

    if save_to:
        rand_str=functools.reduce(lambda carr,x: carr+x, np.random.choice(list(string.ascii_letters+string.digits),size=10).tolist(),"")
        out_path=save_to+"tokenizer_"+rand_str+""
        wrapped_tokenizer.save_pretrained(out_path)
        print(colored(f"wrapped tokenizer was saved to {out_path}","green",attrs=["bold"]))


    return wrapped_tokenizer, tokenizer


if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='tokenization utils')
    _parser.add_argument('--input_archive', type=str,  help="path to input archive", default="",required=True)
    _parser.add_argument('--save_tokenizer_to', type=str,  help="directory where the tokenizer is saved", default="")

    _args=_parser.parse_args()

    with open(_args.input_archive,"rb") as fl:
        _arch=pickle.load(fl)

    _wrapped_tok,_tok=learn_tokenizer_for_archive(_arch,save_to=_args.save_tokenizer_to)

    #sort per id
    _sorted_vocab=sorted([(k,v) for k,v in _tok.get_vocab().items()],key=lambda x:x[1])

    _example="after passing the fridge, go towards the statue, then circle back towards the bed, and then go to the cactus"
    encoding_tok=_tok.encode(_example);
    encoding_wrapped_tok=_wrapped_tok.encode(_example)
    print("example==",_example)
    print("=========== tokenizer outputs =========")
    print(f"encoding tokens=={encoding_tok.tokens}")
    print(f"encoding ids=={encoding_tok.ids}")
    print(f"encoding type_ids=={encoding_tok.type_ids}")
    print(f"decoding=={_tok.decode(encoding_tok.ids)}")
    print("=========== wrapped tokenizer outputs =========")
    print(f"encoding ids=={encoding_wrapped_tok}")
    print(f"decoding=={_wrapped_tok.decode(encoding_wrapped_tok)}")



