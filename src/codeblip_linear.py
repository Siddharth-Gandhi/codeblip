# import logging
# import warnings

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, logging

from codeblip import CodeBlip

logging.set_verbosity_error()  # Only show errors, not warnings

class CodeBlipLinear(CodeBlip):
    def __init__(self, t5_tokenizer, t5_model, prompt, max_source_len=512,
        max_target_len=512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.tokenizer = self.init_tokenizer()
        self.code_encoder, self.ln_code = self.init_code_encoder()

        # freeze the encoder
        for param in self.code_encoder.parameters():
            param.requires_grad = False
        self.code_encoder.eval()

        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model

        for param in self.t5_model.parameters():
            param.requires_grad = False
            # param.data = param.data.float16()
            # make it fp16
            param.data = param.data.half()

        self.t5_proj = nn.Linear(
            self.code_encoder.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.prompt = prompt


    def forward(self, samples):
        source_code = samples["source_code"] # image
        # target_code = samples["target_code"] # text

        # Process source code
        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True,
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.device)

        # code embedding
        source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)

        # pass it through t5_proj
        t5_input = self.t5_proj(source_output)


        output_tokens = self.t5_tokenizer(
            samples['target_code'],
            padding="longest",
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt",
        ).to(self.device)


        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        outputs = self.t5_model(
            inputs_embeds=t5_input,
            attention_mask=source_tokens.attention_mask,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        source_code,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=0.9,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - source_code (list): A list of source code strings.
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each source code.
        Returns:
            captions (list): A list of strings of length num_captions.
        """

        # source_code = samples["source_code"]

        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True,
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.device)

        source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)

        inputs_t5 = self.t5_proj(source_output)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(self.device)

        input_tokens = self.t5_tokenizer(
            self.prompt,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text[0]

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError



if __name__ == '__main__':

    t5_model = 'salesforce/codet5-large'
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CodeBlipLinear(t5_tokenizer, t5_model, prompt="translate Java to C#").to(device)

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

    # Generate captions
    captions = model.generate(source_code_samples[0])
    print(f"Generated caption: {captions}")