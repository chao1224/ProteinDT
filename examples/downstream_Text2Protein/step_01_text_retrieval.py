import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch


def load_text_sequence_file(file_path):
    text_sequence_list = []
    f = open(file_path, 'r')
    for line_idx, line in enumerate(f.readlines()):
        line = line.strip()
        if line_idx % 2 == 0:
            uniprot = line
        else:
            text_sequence = line.strip()           
            text_sequence_list.append(text_sequence)
    return text_sequence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    print("arguments", args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    text_sequence_file = "../../data/SwissProtCLAP/text_sequence.txt"
    text_sequence_list = load_text_sequence_file(text_sequence_file)
    text_sequence_set = set(text_sequence_list)
    print("len of text_sequence_list: {}".format(len(text_sequence_list)))
    print("len of text_sequence_set: {}".format(len(text_sequence_set)))
    # len of text_sequence_list: 541159
    # len of text_sequence_set: 146044
    text_sequence_list = list(text_sequence_list)

    output_file = "step_01_text_retrieval.txt"
    f = open(output_file, 'w')
    sampled_text_sequence_list = random.sample(text_sequence_list, 10000)
    for text_sequence in sampled_text_sequence_list:
        print(text_sequence, file=f)
