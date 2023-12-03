# import logging
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, T5TokenizerFast, logging

from codeblip import CodeBlip
from codeblip_qformer import CodeQformer
from modelling_t5 import T5Config, T5ForConditionalGeneration

logging.set_verbosity_error()  # Only show errors, not warnings

class CodeBlipT5(CodeBlip):
    def __init__(self, stage1_qformer, stage1_query_tokens, num_query_token=32, t5_model='google/flan-t5-xl', prompt="", max_source_len=512,
        max_target_len=512, embed_dim=768, cross_attention_freq=2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.tokenizer = self.init_tokenizer()

        self.code_encoder, self.ln_code = self.init_code_encoder()

        self.code_encoder.to(self.device)
        self.ln_code.to(self.device)

        for param in self.code_encoder.parameters():
            param.requires_grad = False
        self.code_encoder.eval()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, embed_dim, cross_attention_freq
        # )

        self.Qformer = stage1_qformer
        self.query_tokens = stage1_query_tokens

        # not sure if this is needed
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.Qformer.to(self.device)
        # self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            # param.data = param.data.float16()
            # make it fp16
            param.data = param.data.half()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # self.max_txt_len = max_txt_len
        self.prompt = prompt


    def forward(self, samples):
        source_code = samples["source_code"] # image
        target_code = samples["target_code"] # text

        # Process source code
        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True,
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.device)

        # code embedding
        # source_output = self.ln_code(self.code_encoder(source_tokens.input_ids, attention_mask=source_tokens.attention_mask, return_dict=True).last_hidden_state)
        source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)

        # source_output = self.Qformer.bert(**source_tokens, return_dict=True)
        # source_representations = F.normalize(
        #     self.source_proj(source_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        # expand query tokens to batch size
        query_tokens = self.query_tokens.expand(source_output.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=source_output,
            encoder_attention_mask=source_tokens.attention_mask,
            use_cache=True,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(self.device)

        with self.maybe_autocast(dtype=torch.float16):
            input_tokens = self.t5_tokenizer(
                source_code,
                padding="longest",
                truncation=True,
                max_length=self.max_source_len,
                return_tensors="pt",
            ).to(self.device)
            output_tokens = self.t5_tokenizer(
                target_code,
                padding="longest",
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt",
            ).to(self.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
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


    model = CodeBlipT5(stage1_qformer, stage1_query_tokens, t5_model='Salesforce/codet5-large').to('cuda' if torch.cuda.is_available() else 'cpu')
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