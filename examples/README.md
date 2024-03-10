Most of the checkpoints perform quite stable performance, yet here we provide details on exactly which we use in the manuscript. The checkpoint can be found at [this HuggingFace link](https://huggingface.co/chao1224/ProteinDT/tree/main).

## 1 Downstream: Text-to-Protein Generation

|  | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| Autoregressive (T5) | `ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1/downstream_Text2Protein` | Table 1 |
| Multinomial Diffusion (RNN) | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_04_MultinomialDiffusion_RNN_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Text2Protein` | Table 1 |
| Multinomial Diffusion (BERT) | `ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_MultinomialDiffusion_BertBase_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Text2Protein` | Table 1 |


## 2 Downstream: Zero-shot Text-guided Protein Editing

|  | | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- | -- |
| Latent Interpolation | Autoregressive (T5) | `ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0` | Table 2 |
| | Multinomial Diffusion (RNN) | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_04_MultinomialDiffusion_RNN_lr_1e-5_hidden_16_e_10_unconditional_0` | Table 2 |
| | Multinomial Diffusion (BERT) | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_04_MultinomialDiffusion_BertBase_lr_1e-5_hidden_32_e_10_unconditional_0` | Table 2 |
| Latent Optimization | | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_05_AE_1e-3_3` | Table 2 |


## 3 Downstream: Protein Property Prediction

| | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| InfoNCE | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5` | Table 3 |
| EBM-NCE | `ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5` | Table 3 |
