# lines = """

# Galactica & -- & --\\

# ChatGPT & -- & --\\

# % step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1
# AR & BERT & 49.25 & 27.14 & 97.38 & 91.62\\

# % step_04_MultinomialDiffusion_RNN_lr_1e-4_hidden_32_e_10_unconditional_0
# \ProteinSDE{} & RNN & 23.08 & 9.89 & 38.07 & 17.26\\

# % step_04_MultinomialDiffusion_BertBase_lr_1e-4_hidden_32_e_10_unconditional_0
# \ProteinSDE{} & BERT & 45.26 & 24.21 & 46.94 & 29.59\\
# """

lines = """

Galactica & 51.5 & 29.0 & 19.0\\

ChatGPT & 38.5 & 23.0 & 15.5\\

% step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1
% num_repeat: 16	01
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1/downstream_Retrieval/num_repeat_16_use_prior_inference_01/step_02_inference.out
AR & prior & 97.00 & 91.00 & 83.50\\

% num_repeat: 16	01
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1/downstream_Retrieval/num_repeat_16_no_use_prior_inference_01/step_02_inference.out
AR & no_prior & 49.00 & 27.00 & 20.00\\




% step_04_MultinomialDiffusion_RNN_lr_1e-4_hidden_32_e_10_unconditional_0
% num_repeat: 16
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_04_MultinomialDiffusion_RNN_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Retrieval/num_repeat_16_use_prior_simplified/step_02_inference.out
\ProteinSDE{}-RNN & prior & 40.50 & 21.50 & 15.00\\

% num_repeat: 16
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_04_MultinomialDiffusion_RNN_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Retrieval/num_repeat_16_no_use_prior_simplified/step_02_inference.out
\ProteinSDE{}-RNN & no_prior & 24.00 & 10.50 & 5.50\\





% step_04_MultinomialDiffusion_BertBase_lr_1e-4_hidden_32_e_10_unconditional_0
% num_repeat: 16
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_MultinomialDiffusion_BertBase_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Retrieval/num_repeat_16_use_prior_simplified/step_02_inference.out
\ProteinSDE{}-BERT & prior & 51.50 & 25.00 & 13.50\\

% num_repeat: 16
%  ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/step_04_MultinomialDiffusion_BertBase_lr_1e-4_hidden_32_e_10_unconditional_0/downstream_Retrieval/num_repeat_16_no_use_prior_simplified/step_02_inference.out
\ProteinSDE{}-BERT & no_prior & 35.50 & 17.50 & 9.50\\
"""


if __name__ == "__main__":
    baseline_results_list = []
    proteinDT_results_list_without_facilitator = []
    proteinDT_results_list_with_facilitator = []

    for line in lines.split("\n"):
        line = line.strip()
        if line == "":
            continue

        line = line.replace("\\", "")    
        line = line.split("&")

        if line[0].startswith("Galactica"):
            # print(line)
            baseline_results_list.append([line[1], line[2], line[3]])

        elif line[0].startswith("ChatGPT"):
            # print(line)
            baseline_results_list.append([line[1], line[2], line[3]])

        elif "AR" in line[0] or "ProteinSDE" in line[0]:
            if "no_prior" in line[1]:
                proteinDT_results_list_without_facilitator.append([line[2], line[3], line[4]])
            else:
                proteinDT_results_list_with_facilitator.append([line[2], line[3], line[4]])

    T_list = [4, 10, 20]
    for T_idx, T in enumerate(T_list):
        row = "T = {}".format(T)

        for baseline_results in baseline_results_list:
            row = "{} & {}".format(row, baseline_results[T_idx])

        for proteinDT_results in proteinDT_results_list_without_facilitator:
            row = "{} & {}".format(row, proteinDT_results[T_idx])

        for proteinDT_results in proteinDT_results_list_with_facilitator:
            row = "{} & {}".format(row, proteinDT_results[T_idx])
        
        row = "{} \\\\".format(row)
        print(row)