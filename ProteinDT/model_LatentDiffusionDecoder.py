import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from ProteinDT.models.model_SDE import VESDE, VPSDE
from ProteinDT.models.model_Sampler import ReverseDiffusionPredictor, LangevinCorrector
from ProteinDT.models.score_networks import ToyScoreNetwork, RNNScoreNetwork, BertScoreNetwork

EPS = 1e-6


def get_score_fn(SDE_func, score_network, train=True, continuous=True):
    if not train:
        score_network.eval()

    if isinstance(SDE_func, VPSDE):
        def score_fn(x, x_mask, condition, t):
            if continuous:
                score = score_network(protein_seq_repr=x, protein_seq_attention_mask=x_mask, condition=condition)
                std = SDE_func.marginal_prob(x, t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(SDE_func, VESDE):
        def score_fn(x, x_mask, condition, t):
            if continuous:
                score = score_network(protein_seq_repr=x, protein_seq_attention_mask=x_mask, condition=condition)
            else:    
                raise NotImplementedError(f"Discrete not supported")
            score = -score
            return score

    else:
        raise NotImplementedError(f"SDE class {SDE_func.__class__.__name__} not supported.")

    return score_fn


class LatentDiffusionDecoder(nn.Module):
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

        word_embedding_dim = self.hidden_dim
        output_dim = word_embedding_dim

        if score_network_type == "Toy":    
            self.score_network = ToyScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "RNN":
            self.score_network = RNNScoreNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        elif score_network_type == "BertBase":
            config = BertConfig.from_pretrained(
                "bert-base-uncased",
                cache_dir="../data/temp_Bert_base",
                vocab_size=self.num_classes,
                hidden_size=hidden_dim,
                num_attention_heads=8
            )
            self.score_network = BertScoreNetwork(config=config, output_dim=output_dim)

        self.word_embedding_dim = word_embedding_dim
        self.embedding_layer = nn.Linear(self.num_classes, self.word_embedding_dim, bias=False)
        self.decoder_layer = nn.Linear(word_embedding_dim, self.num_classes)
        self.condition_proj_layer = nn.Linear(self.condition_dim, self.word_embedding_dim)

        self.score_fn = get_score_fn(self.SDE_func, self.score_network, train=True, continuous=True)
        self.predictor = ReverseDiffusionPredictor(self.SDE_func, self.score_fn)
        self.corrector = LangevinCorrector(self.SDE_func, self.score_fn, snr=0.1, n_steps=10)
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
    
        protein_seq_repr = self.embedding_layer(protein_seq_onehot)  # (B, max_seq_len, hidden_dim)
        epsilon = torch.randn_like(protein_seq_repr.float())  # (B, max_seq_len, hidden_dim)
        protein_seq_repr = self.embedding_layer(protein_seq_onehot)  # (B, max_seq_len, hidden_dim)
        mean_noise, std_noise = self.SDE_func.marginal_prob(protein_seq_repr, timesteps)  # (B, max_seq_len, hidden_dim), (B)
        protein_seq_repr_noise = mean_noise + std_noise[:, None, None] * epsilon  # (B, max_seq_len, hidden_dim)

        score = self.score_network(protein_seq_repr=protein_seq_repr_noise, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
        score = -score / std_noise[:, None, None]

        total_SDE_loss = torch.square(score * std_noise[:, None, None] + epsilon)  # (B, max_seq_len, hidden_dim)
        masked_SDE_loss = total_SDE_loss * protein_seq_attention_mask.unsqueeze(2)  # (B, max_seq_len, hidden_dim)
        total_SDE_loss = torch.mean(total_SDE_loss)
        masked_SDE_loss = masked_SDE_loss.sum() / protein_seq_attention_mask.sum()
        SDE_loss = total_SDE_loss + masked_SDE_loss

        # regenerate protein_seq_repr
        protein_seq_ids_pred_logit = self.decoder_layer(protein_seq_repr)  # (B, max_seq_len, num_class)
        flattened_logits = protein_seq_ids_pred_logit.view(-1, protein_seq_ids_pred_logit.size(-1))  # (B*max_sequence_len, num_class)
        flattened_ids = protein_seq_input_ids.view(-1)  # (B*max_sequence_len)
        flattened_mask = protein_seq_attention_mask.view(-1)  # (B*max_sequence_len)
        total_decoding_loss = self.CE_criterion(flattened_logits, flattened_ids)  # (B*max_sequence_len)
        masked_decoding_loss = total_decoding_loss * flattened_mask  # (B*max_sequence_len)
        total_decoding_loss = torch.mean(total_decoding_loss)
        masked_decoding_loss = masked_decoding_loss.sum() / flattened_mask.sum()
        decoding_loss = total_decoding_loss + masked_decoding_loss

        return SDE_loss, decoding_loss

    @torch.no_grad()
    def inference(self, condition, max_seq_len, protein_seq_attention_mask):
        B = condition.size()[0]
        device = condition.device
        
        shape = (B, max_seq_len, self.word_embedding_dim)

        X_T = self.SDE_func.prior_sampling(shape).to(device) # (B, max_seq_len, word_embedding_dim)

        EPSILON = 1e-5

        timesteps = torch.linspace(self.SDE_func.T, EPSILON, self.num_diffusion_timesteps, device=device)  # (num_diffusion_timesteps)

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)

        x = X_T
        for i in range(0, self.num_diffusion_timesteps):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            x, x_mean = self.corrector.update_fn(x, protein_seq_attention_mask, condition, vec_t)

            x, x_mean = self.predictor.update_fn(x, protein_seq_attention_mask, condition, vec_t)  # (B, max_seq_len, num_class), (B, max_seq_len, num_class)

        x = self.decoder_layer(x)

        return x
