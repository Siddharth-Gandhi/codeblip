import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from codeblip import CodeBlip
import torch.distributed as dist
from dist_utils import is_dist_avail_and_initialized

# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     # if use distributed training
#     if not is_dist_avail_and_initialized():
#         return tensor
#
#     tensors_gather = [
#         torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output

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

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

    def forward(self, samples):
        source_code = samples["source_code"] # image
        target_code = samples["target_code"] # text

        # Process source code
        source_tokens = self.tokenizer(
            source_code, padding="max_length", truncation=True, 
            max_length=self.max_source_len, return_tensors="pt"
        ).to(self.Qformer.device)

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

        source_features = F.normalize(
            self.source_proj(query_output.last_hidden_state), dim=-1
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
        # sim_matrix = torch.matmul(source_features, target_features.T) / self.temp
        # loss_contrastive = F.cross_entropy(sim_matrix, torch.arange(source_features.size(0), device=sim_matrix.device))
        #
        # return loss_contrastive

        # Contrastive learning
        # image feat = source_features
        # text feat = target_features

        source_features_all = source_features
        target_features_all = target_features

        #
        sim_q2t = torch.matmul(source_features.unsqueeze(1), target_features_all.unsqueeze(-1)).squeeze()

        # source-target similarity
        sim_s2t, _ = sim_q2t.max(-1)
        sim_s2t /= self.temp

        # sim_t2q = torch.matmul(target_features.unsqueeze(1), source_features_all.unsqueeze(-1)).squeeze()
        sim_t2q = torch.matmul(
            target_features.unsqueeze(1).unsqueeze(1), source_features_all.permute(0, 2, 1)
        ).squeeze()
        # target-source similarity ; aggregate the max across all query tokens
        sim_t2s, _ = sim_t2q.max(-1)
        sim_t2s /= self.temp

        # rank = dist.get_rank()
        rank = 0 # something to do with distributed training, but we're not using it so just set to 0
        bs = source_features.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(sim_s2t.device)

        loss_stc = (F.cross_entropy(sim_s2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2s, targets, label_smoothing=0.1)) / 2

        # return loss_stc

        # Sorce - Target Matching
        # image feat = source_features
        # text feat = target_features
        target_input_ids_world = target_tokens.input_ids # if distributed, this is all_gather(target_tokens.input_ids)
        target_attention_mask_world = target_tokens.attention_mask # if distributed, this is all_gather(target_tokens.attention_mask)
        # image_embeds_world = all_gather_with_grad(image_embeds)
        source_features_world = source_features_all # if distributed, this is all_gather_with_grad(source_features_all)
        with torch.no_grad():
            # if "image_id" in samples.keys():
            #     mask = torch.eq(image_ids, image_ids_all.t())
            #     sim_t2i.masked_fill_(mask, -10000)
            #     sim_i2t.masked_fill_(mask, -10000)
            # else:
            sim_s2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
            sim_t2s[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2s = F.softmax(sim_t2s, dim=1)
            weights_s2t = F.softmax(sim_s2t, dim=1)

        # select a negative image for each text
        source_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            source_embeds_neg.append(source_features_world[neg_idx])
        source_embeds_neg = torch.stack(source_embeds_neg, dim=0)

        # select a negative text for each image
        target_ids_neg = []
        target_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            target_ids_neg.append(target_input_ids_world[neg_idx])
            target_atts_neg.append(target_attention_mask_world[neg_idx])

        target_ids_neg = torch.stack(target_ids_neg, dim=0)
        target_atts_neg = torch.stack(target_atts_neg, dim=0)

        target_ids_all = torch.cat(
            [target_tokens.input_ids, target_tokens.input_ids, target_ids_neg], dim=0
        )  # pos, pos, neg
        target_atts_all = torch.cat(
            [target_tokens.attention_mask, target_tokens.attention_mask, target_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(target_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            self.Qformer.device
        )
        attention_mask_all = torch.cat([query_atts_itm, target_atts_all], dim=1)

        source_embeds_all = torch.cat(
            [source_features, source_embeds_neg, source_features], dim=0
        )  # pos, neg, pos
        source_atts_all = torch.ones(source_embeds_all.size()[:-1], dtype=torch.long).to(
            self.Qformer.device
        )

        output_itm = self.Qformer.bert(
            target_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=source_embeds_all,
            encoder_attention_mask=source_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.Qformer.device)
        loss_itm = F.cross_entropy(logits, itm_labels)


        # 3rd loss
        decoder_input_ids = target_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.Qformer.device
        )
        attention_mask = torch.cat([query_atts, target_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return loss_stc + loss_itm + loss_lm



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
    # disable distributed training
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