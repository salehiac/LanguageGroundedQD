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
from typing import List, Any

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

def get_max_sequence_len(data:List[str], wrapped_tokenizer:PreTrainedTokenizerFast):

    zz=_wrapped_tok(data,padding=False).input_ids
    lens=[len(x) for x in zz]
    uu=np.max([len(x) for x in zz])

    return uu, lens

    




if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='tokenization utils')
    _parser.add_argument('--input_archive', type=str,  help="path to input archive", default="",required=True)
    _parser.add_argument('--save_tokenizer_to', type=str,  help="directory where the tokenizer is saved", default="")
    _parser.add_argument('--load_tokenizer_from', type=str,  help="directory path to tokenizer", default="")
    _parser.add_argument('--arch_tokenization_info',action='store_true',help="print info on tokenization of the llm descriptions in the archive, using the learned tokenizer")

    _args=_parser.parse_args()

    with open(_args.input_archive,"rb") as fl:
        _arch=pickle.load(fl)

    if _args.load_tokenizer_from:
        _wrapped_tok=PreTrainedTokenizerFast.from_pretrained(_args.load_tokenizer_from)
    else:
        _wrapped_tok,_tok=learn_tokenizer_for_archive(_arch,save_to=_args.save_tokenizer_to)

    if _args.arch_tokenization_info:
        dd={}
        dd["maximum sequence length (#tokens)"], _lens=get_max_sequence_len([x._llm_descr for x in _arch], wrapped_tokenizer=_wrapped_tok)
        print(dd)

    if 1:
        _thresh=226
        for ii in range(len(_arch)):
           cur_text=_arch[ii]._llm_descr
           zz=_wrapped_tok(cur_text, padding=False).input_ids
           if len(zz)>_thresh:
               print(ii)




