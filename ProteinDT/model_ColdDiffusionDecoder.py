import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from ProteinDT.models.model_SDE import VESDE, VPSDE
from ProteinDT.models.model_Sampler import ReverseDiffusionPredictor, LangevinCorrector
from ProteinDT.models.score_networks import ToyScoreNetwork, RNNScoreNetwork, BertScoreNetwork

EPS = 1e-6


class ColdDiffusionDecoder(nn.Module):
    def __init__(
        self, hidden_dim, condition_dim, beta_min, beta_max, num_diffusion_timesteps, num_classes, score_network_type
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_classes = num_classes

        self.SDE_func = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)

        output_dim = hidden_dim
        if score_network_type == "Toy":
            word_embedding_dim = self.hidden_dim
            self.score_network = ToyScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "RNN":
            word_embedding_dim = self.hidden_dim
            self.score_network = RNNScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "BertBase":
            config = BertConfig.from_pretrained(
                "bert-base-uncased",
                cache_dir="../data/temp_Bert_base",
                vocab_size=self.num_classes,
                hidden_size=hidden_dim,
                num_attention_heads=8
            )
            word_embedding_dim = self.hidden_dim
            self.score_network = BertScoreNetwork(config=config, output_dim=output_dim)

        self.word_embedding_dim = word_embedding_dim
        self.embedding_layer = nn.Linear(self.num_classes, self.word_embedding_dim, bias=False)
        self.decoder_layer = nn.Linear(word_embedding_dim, self.num_classes)
        self.condition_proj_layer = nn.Linear(self.condition_dim, self.word_embedding_dim)

        self.CE_criterion = nn.CrossEntropyLoss(reduction='none')
        return
    
    def forward(self, protein_seq_input_ids, protein_seq_attention_mask, condition):
        B = protein_seq_input_ids.size()[0]
        device = protein_seq_input_ids.device

        # TODO: need double-check range of timesteps
        timesteps = torch.rand(B, device=device) * (1 - EPS) + EPS  # (B)

        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)
        
        protein_seq_onehot = F.one_hot(protein_seq_input_ids, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
        protein_seq_onehot = protein_seq_onehot.float()

        #### cold diffusion can add noise either in the one-hot level
        epsilon = torch.randn_like(protein_seq_onehot.float())  # (B, max_seq_len, num_class)
        mean_noise, std_noise = self.SDE_func.marginal_prob(protein_seq_onehot, timesteps)  # (B, max_seq_len, num_classes), (B)
        protein_seq_onehot_noise = mean_noise + std_noise[:, None, None] * epsilon  # (B, max_seq_len, num_classes)
        protein_seq_repr_noise = self.embedding_layer(protein_seq_onehot_noise)  # (B, max_seq_len, hidden_dim)

        # ##### TODO: or in the embedding level???
        # protein_seq_repr = self.embedding_layer(protein_seq_onehot)  # (B, max_seq_len, hidden_dim)
        # epsilon = torch.randn_like(protein_seq_repr.float())  # (B, max_seq_len, hidden_dim)
        # protein_seq_repr = self.embedding_layer(protein_seq_onehot)  # (B, max_seq_len, hidden_dim)
        # mean_noise, std_noise = self.SDE_func.marginal_prob(protein_seq_repr, timesteps)  # (B, max_seq_len, hidden_dim), (B)
        # protein_seq_repr_noise = mean_noise + std_noise[:, None, None] * epsilon  # (B, max_seq_len, hidden_dim)
        
        score = self.score_network(protein_seq_repr=protein_seq_repr_noise, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim) or (B, max_seq_len, num_class)
        score = self.decoder_layer(score)   # (B*max_sequence_len, num_class)

        flattened_logits = score.view(-1, score.size(-1))  # (B*max_sequence_len, num_class)
        flattened_ids = protein_seq_input_ids.view(-1)  # (B*max_sequence_len)
        flattened_mask = protein_seq_attention_mask.view(-1)  # (B*max_sequence_len)
        total_SDE_loss = self.CE_criterion(flattened_logits, flattened_ids)  # (B*max_sequence_len)
        masked_SDE_loss = total_SDE_loss * flattened_mask  # (B*max_sequence_len)
        total_SDE_loss = torch.mean(total_SDE_loss)
        masked_SDE_loss = total_SDE_loss.sum() / flattened_mask.sum()

        SDE_loss = total_SDE_loss + masked_SDE_loss
        decoding_loss = 0

        return SDE_loss, decoding_loss

    @torch.no_grad()
    def inference(self, condition, max_seq_len, protein_seq_attention_mask):
        B = condition.size()[0]
        device = condition.device
        
        shape = (B, max_seq_len, self.num_classes)

        X_one_hot = self.SDE_func.prior_sampling(shape).to(device) # (B, max_seq_len, word_embedding_dim)

        EPSILON = 1e-5

        timesteps = torch.linspace(self.SDE_func.T, EPSILON, self.num_diffusion_timesteps, device=device)  # (num_diffusion_timesteps)

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)

        x_one_hot_t = X_one_hot
        for i in range(0, self.num_diffusion_timesteps-1):
            x_repr_t = self.embedding_layer(x_one_hot_t)  # (B, max_seq_len, hidden_dim)
            score = self.score_network(protein_seq_repr=x_repr_t, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
            hat_x_one_hot_0 = self.decoder_layer(score)   # (B, max_sequence_len, num_class)

            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=device) * t  # (B)
            t_1 = timesteps[i+1]
            vec_t_1 = torch.ones(shape[0], device=device) * t_1  # (B)
            epsilon = torch.randn_like(hat_x_one_hot_0)  # (B, max_seq_len, num_class)

            mean_noise, std_noise = self.SDE_func.marginal_prob(hat_x_one_hot_0, vec_t)  # (B, max_seq_len, num_classes), (B)
            x_one_hot_t = mean_noise + std_noise[:, None, None] * epsilon  # (B, max_seq_len, num_classes)
            mean_noise, std_noise = self.SDE_func.marginal_prob(hat_x_one_hot_0, vec_t_1)  # (B, max_seq_len, num_classes), (B)
            x_one_hot_t_1 = mean_noise + std_noise[:, None, None] * epsilon  # (B, max_seq_len, num_classes)

            x_one_hot_t = x_one_hot_t - x_one_hot_t - x_one_hot_t_1   # (B, max_sequence_len, num_class)
        x = x_one_hot_t

        return x
