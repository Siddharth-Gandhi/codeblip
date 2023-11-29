import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from codeblip import CodeBlip

class CodeQformer(CodeBlip):  # Inherits from Blip2Base
    """
    CodeBlip Qformer model for code translation task.
    """

    def __init__(
        self,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_source_len=128,
        max_target_len=128,
    ):
        super().__init__()

        self.code_encoder, self.ln_code = self.init_code_encoder()
        # freeze the encoder
        for param in self.code_encoder.parameters():
            param.requires_grad = False
        self.code_encoder.eval()
        # self.code_encoder.train =


        # Initialize the Qformer and Query Tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, embed_dim, cross_attention_freq
        )
        # Tokenizer for encoding source and target code
        self.tokenizer = self.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # not sure what this does
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # Projection layers for source and target code
        self.source_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.target_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # Temperature parameter for contrastive loss
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # Max lengths for source and target tokens
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def forward(self, samples):
        source_code = samples["source_code"] # image
        target_code = samples["target_code"] # text

        # Process source code
        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True, 
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.Qformer.device)

        # code embedding
        # source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)
        # source_attention_mask = source_tokens["attention_mask"]

        # source_output = self.ln_code(self.code_encoder(source_tokens.input_ids, attention_mask=source_tokens.attention_mask, return_dict=True).last_hidden_state)
        source_output = self.ln_code(self.code_encoder(**source_tokens, return_dict=True).last_hidden_state)
        # get attention



        # source_output = self.Qformer.bert(**source_tokens, return_dict=True)
        # source_representations = F.normalize(
        #     self.source_proj(source_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        # query_tokens = self.query_tokens.expand(source_output.shape[0], -1, -1)
        query_tokens = self.query_tokens.expand(source_output.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=source_output,
            encoder_attention_mask=source_tokens.attention_mask,
            use_cache=True,
            return_dict=True,
        )

        source_features = F.normalize(
            self.source_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # Process target code
        target_tokens = self.tokenizer(
            target_code, padding="max_length", truncation=True, 
            max_length=self.max_target_len, return_tensors="pt"
        ).to(self.Qformer.device)
        target_output = self.Qformer.bert(target_tokens.input_ids, attention_mask=target_tokens.attention_mask, return_dict=True)

        target_features = F.normalize(
            self.target_proj(target_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # Contrastive learning
        sim_matrix = torch.matmul(source_features, target_features.T) / self.temp
        loss_contrastive = F.cross_entropy(sim_matrix, torch.arange(source_features.size(0), device=sim_matrix.device))

        return loss_contrastive

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token", 32)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        embed_dim = cfg.get("embed_dim", 256)
        max_source_len = cfg.get("max_source_len", 128)
        max_target_len = cfg.get("max_target_len", 128)

        return cls(
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )


if __name__ == "__main__":
    # test stage1 loss
    # Assuming CodeQformer and Blip2Base are defined as previously discussed
    # Create an instance of CodeQformer
    model = CodeQformer(num_query_token=32, cross_attention_freq=2, embed_dim=768, max_source_len=512,
                        max_target_len=512)

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
    loss = model(samples)
    print(f"Loss from forward pass: {loss.item()}")