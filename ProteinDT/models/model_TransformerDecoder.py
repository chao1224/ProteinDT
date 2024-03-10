import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput


class T5Decoder(nn.Module):
    def __init__(self, hidden_dim, tokenizer, T5_model):
        super().__init__()
        self.num_classes = tokenizer.vocab_size

        if T5_model == "ProtT5":
            self.config = T5Config.from_pretrained("Rostlab/prot_t5_xl_uniref50", chache_dir="../data/temp_pretrained_prot_t5_xl_uniref50")
            self.T5_model = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        elif T5_model == "T5Base":
            self.config = T5Config.from_pretrained(
                "t5-base",
                chache_dir="../data/temp_t5",
                vocab_size=self.num_classes,
            )
            self.T5_model = T5ForConditionalGeneration(self.config)

        self.rep_linear = nn.Linear(hidden_dim, self.config.d_model)
        self.tokenizer = tokenizer
        return
    
    def forward(self, protein_seq_input_ids, protein_seq_attention_mask, condition):
        condition = condition.unsqueeze(1)
        if condition.size(2) != self.config.d_model:
            condition = self.rep_linear(condition)  # (B, 1, d_model)

        labels = protein_seq_input_ids  # (B, d_model)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        model_outputs = self.T5_model(encoder_outputs=(condition, ), labels=labels)
        loss = model_outputs.loss
        return loss, 0

    def inference(
        self, condition, max_seq_len, protein_seq_attention_mask,
        temperature=1.0, k=40, p=0.9, repetition_penalty=1.0, num_return_sequences=1,
        do_sample=True, num_beams=1
    ):
        if condition.dim() == 2:
            condition = condition.unsqueeze(1).float()  # (B, 1, hidden_dim)
        if condition.size(2) != self.config.d_model:
            condition = self.rep_linear(condition)

        enccoder_outputs = BaseModelOutput(last_hidden_state=condition)
        output_ids = self.T5_model.generate(
            encoder_outputs=enccoder_outputs, 
            max_length=max_seq_len,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin
        return output_ids
