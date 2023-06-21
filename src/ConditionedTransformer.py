
import yaml
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import pdb
import pickle
import numpy as np
import pprint

_pp = pprint.PrettyPrinter(indent=4,sort_dicts=False)

class LanguageConditionedTransformer:
    """
    Not a torchmodule as it has frozen parts
    """

    def __init__(self,
            text_embedding_checkpoint:str,
            **kwargs):
        """
        We use a pretrained tokenizer with a pretrained model
        """

        super().__init__()

        self.text_embedding_checkpoint=text_embedding_checkpoint 
        self.tokenizer=AutoTokenizer.from_pretrained(self.text_embedding_checkpoint)
        self.text_embedder=AutoModel.from_pretrained(self.text_embedding_checkpoint)
        self.text_embedder.eval()

    def embed_text(self, batch, features_type:str):
        """
        features_type  can be either "[CLS]", "AVG" or "ALL". 
                       If "ALL", returns an embedding of shape batch_sz*num_tokens*last_hidden_state_dim , with num_tokens including special tokens
                       If 
        """

        with torch.no_grad():
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            output = self.text_embedder(**tokens)
       
        if features_type=="ALL":
            return output.last_hidden_state, tokens
        if features_type=="AVG":
            return output.last_hidden_state.mean(1), tokens
        if features_type=="[CLS]":
            return output.last_hidden_state[:,0,:], tokens


    def predict(self, batch, features_type):

        text_embeddings, _=self.embed_text(batch,features_type)

        pass







def main_train(cfg):

    lct=LanguageConditionedTransformer(cfg["model_cfg"]["text_embedding"]["model_name"])
    batch_sz=cfg["train_cfg"]["batch_size"]
    raise Exception("not implemted")

def main_test(cfg):

    lct=LanguageConditionedTransformer(cfg["model_cfg"]["text_embedding"]["model_name"])
    batch_sz=cfg["test_cfg"]["batch_size"]

    with open(cfg["test_cfg"]["data_path"],"rb") as fl:
        repertoire=pickle.load(fl)
        repertoire=[x for x in repertoire if hasattr(x,"_llm_descr") and x._llm_descr is not None]

    if cfg["test_cfg"]["debug"]["test_text_embedding_random_batch"]:

        batch=np.random.choice(repertoire,size=batch_sz, replace=False)
        batch_text=[x._llm_descr for x in batch]

        emb, tokens=lct.embed_text(batch_text,cfg["model_cfg"]["text_embedding"]["features_type"])
 
        _pp.pprint(batch_text)
        #pdb.set_trace()
        print("embedding shape==",emb.shape)

        return emb


if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='Language Conditioned Transformer')
    _parser.add_argument('--config', type=str,  help="yaml config", default="")

    _args=_parser.parse_args()

    with open(_args.config,"r") as fl:
        _config=yaml.load(fl,Loader=yaml.FullLoader)

    if _config["train_model"]: 
        _out_train=main_train(_config)
    if _config["test_model"]: 
        _out_test=main_test(_config)
