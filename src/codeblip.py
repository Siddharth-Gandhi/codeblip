import contextlib

import torch
import torch.nn as nn
# from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig
from Qformer import BertLMHeadModel

BERT_MODEL = "microsoft/graphcodebert-base"
class CodeBlip(nn.Module):

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = AutoConfig.from_pretrained(BERT_MODEL)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            BERT_MODEL, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


if __name__ == "__main__":
    # test out initialization of Qformer
    num_query_token = 5
    vision_width = 512
    Qformer, query_tokens = CodeBlip.init_Qformer(num_query_token, vision_width)
    print(Qformer)
    print(query_tokens)