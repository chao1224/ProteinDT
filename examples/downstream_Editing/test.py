import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import string
import mdtraj as md

import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from utils import ProteinDataset, ProteinSeqDataset, text_prompt_dict, load_oracle_evaluator, evaluate, analyze

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37



# traj = md.load_pdb("0.pdb")
traj = md.load_pdb("../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/step_06_AE_1e-3_3/downstream_Editing_latent_optimization/alpha_prompt_101_lambda_0.1_num_repeat_16_oracle_text_T_2/input_PDB/0.pdb")
print(traj)

pdb_ss = md.compute_dssp(traj, simplified=True)[0]  # (L, )
print(pdb_ss)
