"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

# from lavis.common.registry import registry
# from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# # from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
# from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
# import transformers

import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, logging, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM

from codeblip import CodeBlip
from codeblip_qformer import CodeQformer
from modelling_t5 import T5Config, T5ForConditionalGeneration

logging.set_verbosity_error()  # Only show errors, not warnings



class Blip2Llama(CodeBlip):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    def __init__(
        self,
        stage1_qformer, 
        stage1_query_tokens,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        num_query_token=32,
        llama_model="/data/datasets/models/huggingface/meta-llama/CodeLlama-7b-Instruct-hf",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        max_source_len=512,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        # transformers_version = version.parse(transformers.__version__)
        # assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.code_encoder, self.ln_code = self.init_code_encoder()

        self.code_encoder.to(self.device)
        self.ln_code.to(self.device)

        for param in self.code_encoder.parameters():
            param.requires_grad = False
        self.code_encoder.eval()

        self.max_source_len = max_source_len


        self.Qformer = stage1_qformer
        self.query_tokens = stage1_query_tokens

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None


        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model, torch_dtype=torch.float16
        )

        self.llama_model.resize_token_embeddings(len(self.tokenizer))

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.llama_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

    def forward(self, samples):
        source_code = samples["source_code"] # image
        target_code = samples["target_code"] # text



        #self.tokenizer = GraphCodeBert Tokenizer
        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True,
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.device)

        source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)

        query_tokens = self.query_tokens.expand(source_output.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=source_output,
            encoder_attention_mask=source_tokens.attention_mask,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(self.device)

        self.llama_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["target_code"]]

        opt_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llama_model.model.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}




if __name__ == '__main__':
    # test initialization

    stage_1_checkpoint = 'models/stage1_out/stage1_best.pt'
    stage1_model = CodeQformer.from_config({'pretrained_path': stage_1_checkpoint})

    stage1_qformer = stage1_model.Qformer
    stage1_query_tokens = stage1_model.query_tokens


    model = Blip2Llama(stage1_qformer, stage1_query_tokens).to('cuda' if torch.cuda.is_available() else 'cpu')
    # Dummy source and target code samples
    source_code_samples = [
        "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
        "public class Test { public static int add(int a, int b) { return a + b; } }"]
    target_code_samples = [
        "class HelloWorld { static void Main(string[] args) { Console.WriteLine(\"Hello, world!\"); } }",
        "class Test { static int Add(int a, int b) { return a + b; } }"]

    # Prepare input for the model
    samples = {"source_code": source_code_samples, "target_code": target_code_samples}

    # Perform a forward pass
    losses = model(samples)
    print(f"Loss from forward pass: {losses['loss'].item()}")