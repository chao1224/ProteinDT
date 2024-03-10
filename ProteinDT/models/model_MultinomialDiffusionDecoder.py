import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from transformers import BertConfig
from ProteinDT.models.score_networks import ToyScoreNetwork, RNNScoreNetwork, BertScoreNetwork

EPS = 1e-6


class MultinomialDiffusion(nn.Module):
    def __init__(
        self, hidden_dim, condition_dim, beta_min, beta_max, num_diffusion_timesteps, mask_id, num_classes, score_network_type
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_classes = num_classes
        self.mask_id = mask_id        

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
                # cache_dir="../data/temp_Bert_base",
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
        timesteps = torch.randint(1, 1+self.num_diffusion_timesteps, (B,), device=device)
        # pt = torch.ones_like(timesteps).float() / self.num_diffusion_timesteps

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)
        
        protein_seq_onehot = F.one_hot(protein_seq_input_ids, num_classes=self.num_classes)  # (B, max_seq_len, num_class)

        x_t, x_0_ignore = protein_seq_input_ids.clone(), protein_seq_input_ids.clone()
        
        mask = torch.rand_like(x_t.float()) < timesteps.float().unsqueeze(-1) / self.num_diffusion_timesteps  # (B, max_seq_len)
        x_t[mask] = self.mask_id  # (B, max_seq_len)
        x_0_ignore[torch.bitwise_not(mask)] = -1  # (B, max_seq_len)

        x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
        x_t_one_hot = x_t_one_hot.float()
        x_t_repr = self.embedding_layer(x_t_one_hot)  # (B, max_seq_len, hidden_dim)
        
        x_0_repr = self.score_network(protein_seq_repr=x_t_repr, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
        x_0_logits = self.decoder_layer(x_0_repr)   # (B*max_sequence_len, num_class)

        flattened_logits = x_0_logits.view(-1, x_0_logits.size(-1))  # (B*max_sequence_len, num_class)
        flattened_ids = protein_seq_input_ids.view(-1)  # (B*max_sequence_len)
        flattened_mask = protein_seq_attention_mask.view(-1)  # (B*max_sequence_len)
        total_SDE_loss = self.CE_criterion(flattened_logits, flattened_ids)  # (B*max_sequence_len)
        masked_SDE_loss = total_SDE_loss * flattened_mask  # (B*max_sequence_len)
        total_SDE_loss = torch.mean(total_SDE_loss)
        masked_SDE_loss = masked_SDE_loss.sum() / flattened_mask.sum()
        # previously:
        # masked_SDE_loss = total_SDE_loss.sum() / flattened_mask.sum()

        SDE_loss = total_SDE_loss + masked_SDE_loss
        decoding_loss = 0

        return SDE_loss, decoding_loss
        
    @torch.no_grad()
    def inference(self, condition, max_seq_len, protein_seq_attention_mask, mode="simplified"):
        if mode == "simplified":
            return self.simplified_inference(condition, max_seq_len, protein_seq_attention_mask)
        elif mode == "weighted":
            return self.weighted_inference(condition, max_seq_len, protein_seq_attention_mask)
        return
        
    @torch.no_grad()
    def simplified_inference(self, condition, max_seq_len, protein_seq_attention_mask):
        """
        credit to https://github.com/samb-t/unleashing-transformers/blob/master/models/absorbing_diffusion.py#L134
        """
        B = condition.size()[0]
        device = condition.device
        
        shape = (B, max_seq_len, self.num_classes)

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)

        x_t = torch.ones((B, max_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        temperature = 1.

        for timestep in reversed(range(1, 1+self.num_diffusion_timesteps)):
            # TODO: need to double-check
            t = torch.full((B,), timestep, device=device).long()  # (B)
            # t = torch.full((B,), self.num_diffusion_timesteps-timestep+1, device=device).long()  # (B)
            
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)  # (B, max_seq_len)
            # don't unmask somwhere already masked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))  # (B, max_seq_len)
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)  # (B, max_seq_len)

            x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
            x_t_one_hot = x_t_one_hot.float()
            x_t_repr = self.embedding_layer(x_t_one_hot)  # (B, max_seq_len, hidden_dim)

            x_0_repr = self.score_network(protein_seq_repr=x_t_repr, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
            x_0_logits = self.decoder_layer(x_0_repr)   # (B, max_sequence_len, num_class)

            x_0_logits /= temperature
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()

            x_t[changes] = x_0_hat[changes]

        x = x_t
        x = F.one_hot(x, num_classes=self.num_classes)

        return x
        
    @torch.no_grad()
    def weighted_inference(self, condition, max_seq_len, protein_seq_attention_mask):
        B = condition.size()[0]
        device = condition.device
        
        shape = (B, max_seq_len, self.num_classes)

        condition = condition.float()
        condition = self.condition_proj_layer(condition)  # (B, max_seq_len, condition_dim) ---> (B, max_seq_len, hidden_dim)

        x_t = torch.ones((B, max_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        temperature = 1.

        def get_Qt(t):
            # TODO: need to double-check this
            # beta_t = 1. / (self.num_diffusion_timesteps - t + 1)
            beta_t = 1. / t
            Q = torch.eye(self.num_classes) * (1 - beta_t)
            Q[:, self.mask_id] = beta_t
            Q[self.mask_id, self.mask_id] = 1
            Q = Q.to(device)
            return Q

        bar_Q_t = torch.eye(self.num_classes).to(device)

        def posterior(x_t, x_0, t, bar_Q_t):
            """
            q(x_t-1 | x_t, x_0)
            x_t: (B, max_seq_le, vocab_size)
            x_0: (B, max_seq_le, vocab_size)
            """
            Q_t = get_Qt(t)  # (vocab_size, vocab_size)
            bar_Q_t_minus_1 = bar_Q_t  # (vocab_size, vocab_size)
            bar_Q_t = bar_Q_t_minus_1 * Q_t  # (vocab_size, vocab_size)
            fact_1 = torch.matmul(x_t, Q_t.transpose(0,1))  # (B, max_seq_le, vocab_size)
            fact_2 = torch.matmul(x_0, bar_Q_t_minus_1.transpose(0,1))  # (B, max_seq_le, vocab_size)
            denominator = torch.matmul(x_0, bar_Q_t) * x_t  # (B, max_seq_le, vocab_size)
            denominator = denominator.sum(dim=-1, keepdim=True)  # (B, max_seq_le, 1)
            logits = torch.exp(torch.log(fact_1+EPS) + torch.log(fact_2+EPS) - torch.log(denominator+EPS))  # (B, max_seq_le, vocab_size)
            return logits, bar_Q_t

        softmax = torch.nn.Softmax(-1)

        for timestep in reversed(range(1, 1+self.num_diffusion_timesteps)):
            # t = torch.full((B,), timestep, device=device).long()  # (B)
            t = torch.full((B,), self.num_diffusion_timesteps-timestep+1, device=device).long()  # (B)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)  # (B, max_seq_len)
            # don't unmask somwhere already masked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))  # (B, max_seq_len)
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)  # (B, max_seq_len)

            x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes)  # (B, max_seq_len, num_class)
            x_t_one_hot = x_t_one_hot.float()
            x_t_repr = self.embedding_layer(x_t_one_hot)  # (B, max_seq_len, hidden_dim)

            x_0_repr = self.score_network(protein_seq_repr=x_t_repr, protein_seq_attention_mask=protein_seq_attention_mask, condition=condition)  # (B, max_seq_len, hidden_dim)
            x_0_logits = self.decoder_layer(x_0_repr)   # (B, max_sequence_len, num_class)

            x_0_logits /= temperature
            x_0_prob = softmax(x_0_logits)

            posterior_logits, bar_Q_t = posterior(x_t_one_hot, x_0_prob, timestep, bar_Q_t)

            x_0_prob = x_0_prob * posterior_logits
            
            x_0_dist = dists.Categorical(probs=x_0_prob)
            x_0_hat = x_0_dist.sample().long()

            x_t[changes] = x_0_hat[changes]

        x = x_t
        x = F.one_hot(x, num_classes=self.num_classes)

        return x
