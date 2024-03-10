import os
import itertools


def extract(file_path):
    evaluation_result_list = []
    f = open(file_path, 'r')
    for line in f.readlines():
        # print(line)
        if not line.startswith("evaluation_T:"):
            continue
        line = line.replace("accuracy:", ",").replace(":", ",").split(",")
        evaluation_T = int(line[1])
        accuracy = float(line[2])
        evaluation_result_list.append([evaluation_T, accuracy])
    
    row = ""
    for evaluation_result in evaluation_result_list:
        row = "{} & {:.2f}".format(row, evaluation_result[1])

    if "MultinomialDiffusion_RNN" in file_path:
        head = "\\ProteinSDE{}-RNN"
    elif "MultinomialDiffusion_BertBase" in file_path:
        head = "\\ProteinSDE{}-BERT"
    else:
        head = "AR"

    if "no_use_facilitator" in file_path:
        facilitator = "no_facilitator"
    else:
        facilitator = "facilitator"
    
    row = head + " & {}".format(facilitator) + row + "\\\\"
    print(row)
    print()
    return


def extract_results_Diffusion(pretrained_mode, decoder_distribution, score_network_type):
    for lr, hidden_dim, epochs, prob_unconditional in itertools.product(*[lr_list, hidden_dim_list, epochs_list, prob_unconditional_list]):

        step_04_folder = "../../output/{}/step_04_{}_{}_lr_{lr}_hidden_{hidden_dim}_e_{epochs}_unconditional_{prob_unconditional}".format(
            pretrained_mode, decoder_distribution, score_network_type, lr=lr, hidden_dim=hidden_dim, epochs=epochs, prob_unconditional=prob_unconditional)
        # print("step_04_folder", step_04_folder)
        print("% step_04_{}_{}_lr_{lr}_hidden_{hidden_dim}_e_{epochs}_unconditional_{prob_unconditional}".format(decoder_distribution, score_network_type, lr=lr, hidden_dim=hidden_dim, epochs=epochs, prob_unconditional=prob_unconditional))

        for num_repeat, facilitator, SDE_sampling_mode in itertools.product(*[num_repeat_list, facilitator_list, SDE_sampling_mode_list]):
            print("% num_repeat: {}".format(num_repeat))
            retrieval_folder = os.path.join(step_04_folder, "downstream_Retrieval/num_repeat_{}_{}_{}".format(num_repeat, facilitator, SDE_sampling_mode))
            retrieval_file_path = os.path.join(retrieval_folder, "step_02_inference.out")

            try:
                print("% ", retrieval_file_path)
                extract(retrieval_file_path)
            except:
                print("file {} missing or still running.".format(retrieval_file_path))
                continue

        print()

    return


def extract_results_AR(pretrained_mode, decoder_distribution, score_network_type):
    for lr, hidden_dim, epochs, prob_unconditional in itertools.product(*[lr_list, hidden_dim_list, epochs_list, prob_unconditional_list]):

        step_04_folder = "../../output/{}/step_04_{}_{}_lr_{lr}_hidden_{hidden_dim}_e_{epochs}_unconditional_{prob_unconditional}".format(
            pretrained_mode, decoder_distribution, score_network_type, lr=lr, hidden_dim=hidden_dim, epochs=epochs, prob_unconditional=prob_unconditional)
        # print("step_04_folder", step_04_folder)
        print("% step_04_{}_{}_lr_{lr}_hidden_{hidden_dim}_e_{epochs}_unconditional_{prob_unconditional}".format(decoder_distribution, score_network_type, lr=lr, hidden_dim=hidden_dim, epochs=epochs, prob_unconditional=prob_unconditional))

        for num_repeat, facilitator, AR_generation_mode in itertools.product(*[num_repeat_list, facilitator_list, AR_generation_mode_list]):
            print("% num_repeat: {}\t{}".format(num_repeat, AR_generation_mode))
            retrieval_folder = os.path.join(step_04_folder, "downstream_Retrieval/num_repeat_{}_{}_inference_{}".format(num_repeat, facilitator, AR_generation_mode))
            # retrieval_file_path = os.path.join(retrieval_folder, "downstream_Retrieval_step_02_inference.out")
            retrieval_file_path = os.path.join(retrieval_folder, "step_02_inference.out")
            
            try:
                print("% ", retrieval_file_path)
                extract(retrieval_file_path)
            except:
                print("file {} missing or still running.".format(retrieval_file_path))
                continue

        print()

    return


if __name__ == "__main__":
    # Hyperparameters for step-04 pretraining
    pretrained_mode_list = [
        "ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10",
        "ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5",
        "ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5",
    ]
    prob_unconditional_list = [0, 0.1]
    epochs_list = [10]
    hidden_dim_list = [16, 32]
    facilitator_list = ["use_facilitator", "no_use_facilitator"]
    SDE_sampling_mode_list =["simplified", "weighted"]

    decoder_distribution = "MultinomialDiffusion"
    score_network_type = "RNN"
    lr_list = ["1e-4", "1e-5"]
    num_repeat_list = [16, 32]
    for pretrained_mode in pretrained_mode_list:
        extract_results_Diffusion(pretrained_mode, decoder_distribution, score_network_type)
        print("\n\n\n")
    print("\n\n\n")

    decoder_distribution = "MultinomialDiffusion"
    score_network_type = "BertBase"
    lr_list = ["1e-4", "1e-5"]
    num_repeat_list = [16, 32]
    for pretrained_mode in pretrained_mode_list:
        extract_results_Diffusion(pretrained_mode, decoder_distribution, score_network_type)
        print("\n\n\n")
    print("\n\n\n")

    # epochs_list = [10, 50]
    epochs_list = [10]
    decoder_distribution = "T5Decoder"
    score_network_type = "T5Base"
    lr_list = ["1e-4", "1e-5"]
    # num_repeat_list = [16, 8]
    num_repeat_list = [16]
    AR_generation_mode_list = ["01", "02"]
    for pretrained_mode in pretrained_mode_list:
        extract_results_AR(pretrained_mode, decoder_distribution, score_network_type)
        print("\n\n\n")
    print("\n\n\n")
