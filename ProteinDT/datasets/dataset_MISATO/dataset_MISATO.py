import os
import numpy as np
import h5py
from tqdm import tqdm
import pickle
from Bio.PDB.Polypeptide import three_to_one, is_aa

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

utils_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "utils")


def update_residue_indices(i, type_string, protein_atom_index, residue_index, residue_name, current_residue_index, current_atom_in_residue_index, residue_Map, protein_atom_index2standard_name_dict, molecules_begin_atom_index):
    if i < len(protein_atom_index)-1:
        if type_string == 'O' and protein_atom_index2standard_name_dict[protein_atom_index[i+1]] == 'N' or residue_Map[residue_index[i+1]]=='MOL':
            # GLN has a O N sequence within the AA
            if not ((residue_name == 'GLN' and current_atom_in_residue_index==12) or (residue_name == 'ASN' and current_atom_in_residue_index==9)):
                current_residue_index +=1
                current_atom_in_residue_index = 0
    
    if i+1 in molecules_begin_atom_index:
        current_residue_index +=1
        current_atom_in_residue_index = 0

    return current_residue_index, current_atom_in_residue_index


def get_aa_index(residue):
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, "X":20}
    
    if not is_aa(residue, standard=True):
        return 'X'
        # return 20
    
    one_letter = three_to_one(residue)

    return one_letter
    # return letter_to_num[one_letter]


def get_atom_and_residue_type(protein_atom_index, residue_index, atom_index, protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, molecules_begin_atom_index):
    current_residue_index = 1
    current_atom_in_residue_index = 0
    standard_atom_name_list = []
    current_atom_in_residue_list = []
    residue_type_list = []
    
    L = len(protein_atom_index)
    for i in range(L):
        current_atom_in_residue_index += 1
        atom_type_string = protein_atom_index2standard_name_dict[protein_atom_index[i]]
        residue_name_string = residue_index2name_dict[residue_index[i]]
        try:
            standard_atom_name_string = atom_reisdue2standard_atom_name_dict[(residue_name_string, current_atom_in_residue_index-1, atom_type_string)]
        except KeyError:
            # standard atom name: atomic_number + current_atom_in_residue_index
            standard_atom_name_string = atom_index2name_dict[atom_index[i]] + str(current_atom_in_residue_index)

        neo_current_residue_index, current_atom_in_residue_index = update_residue_indices(i, atom_type_string, protein_atom_index, residue_index, residue_name_string, current_residue_index, current_atom_in_residue_index, residue_index2name_dict, protein_atom_index2standard_name_dict, molecules_begin_atom_index)

        if neo_current_residue_index != current_residue_index:
            current_residue_index = neo_current_residue_index

            current_atom_in_residue_list = np.array(current_atom_in_residue_list)
            residue_type_list.append(get_aa_index(residue_name_string))
            if np.count_nonzero(current_atom_in_residue_list == "CA") == np.count_nonzero(current_atom_in_residue_list == "C") and np.count_nonzero(current_atom_in_residue_list == "N") == np.count_nonzero(current_atom_in_residue_list == "C"):
                standard_atom_name_list.append(current_atom_in_residue_list)
            else:
                standard_atom_name_list.append(np.full(len(current_atom_in_residue_list), 'X', dtype=str))
            current_atom_in_residue_list = []
        
        current_atom_in_residue_list.append(standard_atom_name_string)

    # handle the last residue
    current_atom_in_residue_list = np.array(current_atom_in_residue_list)
    residue_type_list.append(get_aa_index(residue_name_string))
    if np.count_nonzero(current_atom_in_residue_list == "CA") == np.count_nonzero(current_atom_in_residue_list == "C") and np.count_nonzero(current_atom_in_residue_list == "N") == np.count_nonzero(current_atom_in_residue_list == "C"):
        standard_atom_name_list.append(current_atom_in_residue_list)
    else:
        standard_atom_name_list.append(np.full(len(current_atom_in_residue_list), 'X', dtype=str))

    flattened_standard_atom_name_list = [atom for residue in standard_atom_name_list for atom in residue]
    return np.array(flattened_standard_atom_name_list), residue_type_list


def extract_sequence_and_structure(protein_atom_index, residue_index, atom_index, protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, molecules_begin_atom_index):
    """
    protein_atom_index2standard_name_dict: {1: '2C', 2: '3C', 3: 'C', 4: 'C*', 5: 'C8', 6: 'CA', 7: 'CB', 8: 'CC', 9: 'CN', 10: 'CO', 11: 'CR', 12: 'CT', 13: 'CW', 14: 'CX', 15: 'H', 16: 'H1', 17: 'H4', 18: 'H5', 19: 'HA', 20: 'HC', 21: 'HO', 22: 'HP', 23: 'HS', 24: 'N', 25: 'N2', 26: 'N3', 27: 'NA', 28: 'NB', 29: 'O', 30: 'O2', 31: 'OH', 32: 'S', 33: 'SH', 34: 'br', 35: 'c', 36: 'c1', 37: 'c2', 38: 'c3', 39: 'ca', 40: 'cc', 41: 'cd', 42: 'ce', 43: 'cf', 44: 'cg', 45: 'ch', 46: 'cl', 47: 'cp', 48: 'cq', 49: 'cs', 50: 'cu', 51: 'cx', 52: 'cy', 53: 'cz', 54: 'f', 55: 'h1', 56: 'h2', 57: 'h3', 58: 'h4', 59: 'h5', 60: 'ha', 61: 'hc', 62: 'hn', 63: 'ho', 64: 'hp', 65: 'hs', 66: 'hx', 67: 'i', 68: 'n', 69: 'n1', 70: 'n2', 71: 'n3', 72: 'n4', 73: 'n7', 74: 'n8', 75: 'na', 76: 'nb', 77: 'nc', 78: 'nd', 79: 'ne', 80: 'nf', 81: 'nh', 82: 'ni', 83: 'nj', 84: 'nk', 85: 'nl', 86: 'nm', 87: 'nn', 88: 'no', 89: 'nq', 90: 'ns', 91: 'nt', 92: 'nu', 93: 'nv', 94: 'nx', 95: 'ny', 96: 'nz', 97: 'o', 98: 'oh', 99: 'op', 100: 'oq', 101: 'os', 102: 'p5', 103: 'py', 104: 's', 105: 's4', 106: 's6', 107: 'sh', 108: 'ss', 109: 'sx', 110: 'sy'}
    
    residue_index2name_dict {0: 'MOL', 1: 'ACE', 2: 'ALA', 3: 'ARG', 4: 'ASN', 5: 'ASP', 6: 'CYS', 7: 'CYX', 8: 'GLN', 9: 'GLU', 10: 'GLY', 11: 'HIE', 12: 'ILE', 13: 'LEU', 14: 'LYS', 15: 'MET', 16: 'PHE', 17: 'PRO', 18: 'SER', 19: 'THR', 20: 'TRP', 21: 'TYR', 22: 'VAL'}
    
    atom_reisdue2standard_atom_name_dict {('PRO', 0, 'N3'): 'N', ('PRO', 1, 'H'): 'H2', ('PRO', 2, 'H'): 'H3', ('PRO', 3, 'CT'): 'CD', ('PRO', 4, 'HP'): 'HD2', ('PRO', 5, 'HP'): 'HD3', ('PRO', 6, 'CT'): 'CG', ('PRO', 7, 'HC'): 'HG2', ('PRO', 8, 'HC'): 'HB2', ('PRO', 9, 'CT'): 'CB', ('PRO', 10, 'HC'): 'HB2', ('PRO', 11, 'HC'): 'HB3', ('PRO', 12, 'CX'): 'CA', ('PRO', 13, 'HP'): 'HA', ('PRO', 14, 'C'): 'C', ('PRO', 15, 'O'): 'O', ('TYR', 0, 'N'): 'N', ('TYR', 1, 'H'): 'H', ('TYR', 2, 'CX'): 'CA', ('TYR', 3, 'H1'): 'HA', ('TYR', 4, 'CT'): 'CB', ('TYR', 5, 'HC'): 'HB2', ('TYR', 6, 'HC'): 'HB3', ('TYR', 7, 'CA'): 'CG', ('TYR', 8, 'CA'): 'CD1', ('TYR', 9, 'HA'): 'HD1', ('TYR', 10, 'CA'): 'CE1', ('TYR', 11, 'HA'): 'HE1', ('TYR', 12, 'C'): 'CZ', ('TYR', 13, 'OH'): 'OH', ('TYR', 14, 'HO'): 'HH', ('TYR', 15, 'CA'): 'CE2', ('TYR', 16, 'HA'): 'HE2', ('TYR', 17, 'CA'): 'CD2', ('TYR', 18, 'HA'): 'HD2', ('TYR', 19, 'C'): 'C', ('TYR', 20, 'O'): 'O', ('THR', 0, 'N'): 'N', ('THR', 1, 'H'): 'H', ('THR', 2, 'CX'): 'CA', ('THR', 3, 'H1'): 'HA', ('THR', 4, '3C'): 'CB', ('THR', 5, 'H1'): 'HB', ('THR', 6, 'CT'): 'CG2', ('THR', 7, 'HC'): 'HG21', ('THR', 8, 'HC'): 'HG22', ('THR', 9, 'HC'): 'HG23', ('THR', 10, 'OH'): 'OG1', ('THR', 11, 'HO'): 'HG1', ('THR', 12, 'C'): 'C', ('THR', 13, 'O'): 'O', ('VAL', 0, 'N'): 'N', ('VAL', 1, 'H'): 'H', ('VAL', 2, 'CX'): 'CA', ('VAL', 3, 'H1'): 'HA', ('VAL', 4, '3C'): 'CB', ('VAL', 5, 'HC'): 'HB', ('VAL', 6, 'CT'): 'CG1', ('VAL', 7, 'HC'): 'HG11', ('VAL', 8, 'HC'): 'HG12', ('VAL', 9, 'HC'): 'HG13', ('VAL', 10, 'CT'): 'CG2', ('VAL', 11, 'HC'): 'HG21', ('VAL', 12, 'HC'): 'HG22', ('VAL', 13, 'HC'): 'HG23', ('VAL', 14, 'C'): 'C', ('VAL', 15, 'O'): 'O', ('PHE', 0, 'N'): 'N', ('PHE', 1, 'H'): 'H', ('PHE', 2, 'CX'): 'CA', ('PHE', 3, 'H1'): 'HA', ('PHE', 4, 'CT'): 'CB', ('PHE', 5, 'HC'): 'HB2', ('PHE', 6, 'HC'): 'HB3', ('PHE', 7, 'CA'): 'CG', ('PHE', 8, 'CA'): 'CD1', ('PHE', 9, 'HA'): 'HD1', ('PHE', 10, 'CA'): 'CE1', ('PHE', 11, 'HA'): 'HE1', ('PHE', 12, 'CA'): 'CZ', ('PHE', 13, 'HA'): 'HZ', ('PHE', 14, 'CA'): 'CE2', ('PHE', 15, 'HA'): 'HE2', ('PHE', 16, 'CA'): 'CD2', ('PHE', 17, 'HA'): 'HD2', ('PHE', 18, 'C'): 'C', ('PHE', 19, 'O'): 'O', ('PRO', 0, 'N'): 'N', ('PRO', 1, 'CT'): 'CD', ('PRO', 2, 'H1'): 'HD2', ('PRO', 3, 'H1'): 'HD3', ('PRO', 4, 'CT'): 'CG', ('PRO', 5, 'HC'): 'HG2', ('PRO', 6, 'HC'): 'HG3', ('PRO', 7, 'CT'): 'CB', ('PRO', 9, 'HC'): 'HB3', ('PRO', 10, 'CX'): 'CA', ('PRO', 11, 'H1'): 'HA', ('PRO', 12, 'C'): 'C', ('PRO', 13, 'O'): 'O', ('ARG', 0, 'N'): 'N', ('ARG', 1, 'H'): 'H', ('ARG', 2, 'CX'): 'CA', ('ARG', 3, 'H1'): 'HA', ('ARG', 4, 'C8'): 'CB', ('ARG', 5, 'HC'): 'HB2', ('ARG', 6, 'HC'): 'HB3', ('ARG', 7, 'C8'): 'CG', ('ARG', 8, 'HC'): 'HG2', ('ARG', 9, 'HC'): 'HG3', ('ARG', 10, 'C8'): 'CD', ('ARG', 11, 'H1'): 'HD2', ('ARG', 12, 'H1'): 'HD3', ('ARG', 13, 'N2'): 'NE', ('ARG', 14, 'H'): 'HE', ('ARG', 15, 'CA'): 'CZ', ('ARG', 16, 'N2'): 'NH1', ('ARG', 17, 'H'): 'HH11', ('ARG', 18, 'H'): 'HH12', ('ARG', 19, 'N2'): 'NH2', ('ARG', 20, 'H'): 'HH21', ('ARG', 21, 'H'): 'HH22', ('ARG', 22, 'C'): 'C', ('ARG', 23, 'O'): 'O', ('GLY', 0, 'N'): 'N', ('GLY', 1, 'H'): 'H', ('GLY', 2, 'CX'): 'CA', ('GLY', 3, 'H1'): 'HA2', ('GLY', 4, 'H1'): 'HA3', ('GLY', 5, 'C'): 'C', ('GLY', 6, 'O'): 'O', ('CYS', 0, 'N'): 'N', ('CYS', 1, 'H'): 'H', ('CYS', 2, 'CX'): 'CA', ('CYS', 3, 'H1'): 'HA', ('CYS', 4, '2C'): 'CB', ('CYS', 5, 'H1'): 'HB2', ('CYS', 6, 'H1'): 'HB3', ('CYS', 7, 'SH'): 'SG', ('CYS', 8, 'HS'): 'HG', ('CYS', 9, 'C'): 'C', ('CYS', 10, 'O'): 'O', ('ALA', 0, 'N'): 'N', ('ALA', 1, 'H'): 'H', ('ALA', 2, 'CX'): 'CA', ('ALA', 3, 'H1'): 'HA', ('ALA', 4, 'CT'): 'CB', ('ALA', 5, 'HC'): 'HB1', ('ALA', 6, 'HC'): 'HB2', ('ALA', 7, 'HC'): 'HB3', ('ALA', 8, 'C'): 'C', ('ALA', 9, 'O'): 'O', ('LEU', 0, 'N'): 'N', ('LEU', 1, 'H'): 'H', ('LEU', 2, 'CX'): 'CA', ('LEU', 3, 'H1'): 'HA', ('LEU', 4, '2C'): 'CB', ('LEU', 5, 'HC'): 'HB2', ('LEU', 6, 'HC'): 'HB3', ('LEU', 7, '3C'): 'CG', ('LEU', 8, 'HC'): 'HG', ('LEU', 9, 'CT'): 'CD1', ('LEU', 10, 'HC'): 'HD11', ('LEU', 11, 'HC'): 'HD12', ('LEU', 12, 'HC'): 'HD13', ('LEU', 13, 'CT'): 'CD2', ('LEU', 14, 'HC'): 'HD21', ('LEU', 15, 'HC'): 'HD22', ('LEU', 16, 'HC'): 'HD23', ('LEU', 17, 'C'): 'C', ('LEU', 18, 'O'): 'O', ('MET', 0, 'N'): 'N', ('MET', 1, 'H'): 'H', ('MET', 2, 'CX'): 'CA', ('MET', 3, 'H1'): 'HA', ('MET', 4, '2C'): 'CB', ('MET', 5, 'HC'): 'HB2', ('MET', 6, 'HC'): 'HB3', ('MET', 7, '2C'): 'CG', ('MET', 8, 'H1'): 'HG2', ('MET', 9, 'H1'): 'HG3', ('MET', 10, 'S'): 'SD', ('MET', 11, 'CT'): 'CE', ('MET', 12, 'H1'): 'HE1', ('MET', 13, 'H1'): 'HE2', ('MET', 14, 'H1'): 'HE3', ('MET', 15, 'C'): 'C', ('MET', 16, 'O'): 'O', ('ASP', 0, 'N'): 'N', ('ASP', 1, 'H'): 'H', ('ASP', 2, 'CX'): 'CA', ('ASP', 3, 'H1'): 'HA', ('ASP', 4, '2C'): 'CB', ('ASP', 5, 'HC'): 'HB2', ('ASP', 6, 'HC'): 'HB3', ('ASP', 7, 'CO'): 'CG', ('ASP', 8, 'O2'): 'OD1', ('ASP', 9, 'O2'): 'OD2', ('ASP', 10, 'C'): 'C', ('ASP', 11, 'O'): 'O', ('GLN', 0, 'N'): 'N', ('GLN', 1, 'H'): 'H', ('GLN', 2, 'CX'): 'CA', ('GLN', 3, 'H1'): 'HA', ('GLN', 4, '2C'): 'CB', ('GLN', 5, 'HC'): 'HB2', ('GLN', 6, 'HC'): 'HB3', ('GLN', 7, '2C'): 'CG', ('GLN', 8, 'HC'): 'HG2', ('GLN', 9, 'HC'): 'HG3', ('GLN', 10, 'C'): 'CD', ('GLN', 11, 'O'): 'OE1', ('GLN', 12, 'N'): 'NE2', ('GLN', 13, 'H'): 'HE21', ('GLN', 14, 'H'): 'HE22', ('GLN', 15, 'C'): 'C', ('GLN', 16, 'O'): 'O', ('SER', 0, 'N'): 'N', ('SER', 1, 'H'): 'H', ('SER', 2, 'CX'): 'CA', ('SER', 3, 'H1'): 'HA', ('SER', 4, '2C'): 'CB', ('SER', 5, 'H1'): 'HB2', ('SER', 6, 'H1'): 'HB3', ('SER', 7, 'OH'): 'OG', ('SER', 8, 'HO'): 'HG', ('SER', 9, 'C'): 'C', ('SER', 10, 'O'): 'O', ('TRP', 0, 'N'): 'N', ('TRP', 1, 'H'): 'H', ('TRP', 2, 'CX'): 'CA', ('TRP', 3, 'H1'): 'HA', ('TRP', 4, 'CT'): 'CB', ('TRP', 5, 'HC'): 'HB2', ('TRP', 6, 'HC'): 'HB3', ('TRP', 7, 'C*'): 'CG', ('TRP', 8, 'CW'): 'CD1', ('TRP', 9, 'H4'): 'HD1', ('TRP', 10, 'NA'): 'NE1', ('TRP', 11, 'H'): 'HE1', ('TRP', 12, 'CN'): 'CE2', ('TRP', 13, 'CA'): 'CZ2', ('TRP', 14, 'HA'): 'HZ2', ('TRP', 15, 'CA'): 'CH2', ('TRP', 16, 'HA'): 'HH2', ('TRP', 17, 'CA'): 'CZ3', ('TRP', 18, 'HA'): 'HZ3', ('TRP', 19, 'CA'): 'CE3', ('TRP', 20, 'HA'): 'HE3', ('TRP', 21, 'CB'): 'CD2', ('TRP', 22, 'C'): 'C', ('TRP', 23, 'O'): 'O', ('LYS', 0, 'N'): 'N', ('LYS', 1, 'H'): 'H', ('LYS', 2, 'CX'): 'CA', ('LYS', 3, 'H1'): 'HA', ('LYS', 4, 'C8'): 'CB', ('LYS', 5, 'HC'): 'HB2', ('LYS', 6, 'HC'): 'HB3', ('LYS', 7, 'C8'): 'CG', ('LYS', 8, 'HC'): 'HG2', ('LYS', 9, 'HC'): 'HG3', ('LYS', 10, 'C8'): 'CD', ('LYS', 11, 'HC'): 'HD2', ('LYS', 12, 'HC'): 'HD3', ('LYS', 13, 'C8'): 'CE', ('LYS', 14, 'HP'): 'HE2', ('LYS', 15, 'HP'): 'HE3', ('LYS', 16, 'N3'): 'NZ', ('LYS', 17, 'H'): 'HZ1', ('LYS', 18, 'H'): 'HZ2', ('LYS', 19, 'H'): 'HZ3', ('LYS', 20, 'C'): 'C', ('LYS', 21, 'O'): 'O', ('GLU', 0, 'N'): 'N', ('GLU', 1, 'H'): 'H', ('GLU', 2, 'CX'): 'CA', ('GLU', 3, 'H1'): 'HA', ('GLU', 4, '2C'): 'CB', ('GLU', 5, 'HC'): 'HB2', ('GLU', 6, 'HC'): 'HB3', ('GLU', 7, '2C'): 'CG', ('GLU', 8, 'HC'): 'HG2', ('GLU', 9, 'HC'): 'HG3', ('GLU', 10, 'CO'): 'CD', ('GLU', 11, 'O2'): 'OE1', ('GLU', 12, 'O2'): 'OE2', ('GLU', 13, 'C'): 'C', ('GLU', 14, 'O'): 'O', ('ASN', 0, 'N'): 'N', ('ASN', 1, 'H'): 'H', ('ASN', 2, 'CX'): 'CA', ('ASN', 3, 'H1'): 'HA', ('ASN', 4, '2C'): 'CB', ('ASN', 5, 'HC'): 'HB2', ('ASN', 6, 'HC'): 'HB3', ('ASN', 7, 'C'): 'CG', ('ASN', 8, 'O'): 'OD1', ('ASN', 9, 'N'): 'ND2', ('ASN', 10, 'H'): 'HD21', ('ASN', 11, 'H'): 'HD22', ('ASN', 12, 'C'): 'C', ('ASN', 13, 'O'): 'O', ('ILE', 0, 'N'): 'N', ('ILE', 1, 'H'): 'H', ('ILE', 2, 'CX'): 'CA', ('ILE', 3, 'H1'): 'HA', ('ILE', 4, '3C'): 'CB', ('ILE', 5, 'HC'): 'HB', ('ILE', 6, 'CT'): 'CG2', ('ILE', 7, 'HC'): 'HG21', ('ILE', 8, 'HC'): 'HG22', ('ILE', 9, 'HC'): 'HG23', ('ILE', 10, '2C'): 'CG1', ('ILE', 11, 'HC'): 'HG12', ('ILE', 12, 'HC'): 'HG13', ('ILE', 13, 'CT'): 'CD1', ('ILE', 14, 'HC'): 'HD11', ('ILE', 15, 'HC'): 'HD12', ('ILE', 16, 'HC'): 'HD13', ('ILE', 17, 'C'): 'C', ('ILE', 18, 'O'): 'O', ('HIE', 0, 'N'): 'N', ('HIE', 1, 'H'): 'H', ('HIE', 2, 'CX'): 'CA', ('HIE', 3, 'H1'): 'HA', ('HIE', 4, 'CT'): 'CB', ('HIE', 5, 'HC'): 'HB2', ('HIE', 6, 'HC'): 'HB3', ('HIE', 7, 'CC'): 'CG', ('HIE', 8, 'NB'): 'ND1', ('HIE', 9, 'CR'): 'CE1', ('HIE', 10, 'H5'): 'HE1', ('HIE', 11, 'NA'): 'NE2', ('HIE', 12, 'H'): 'HE2', ('HIE', 13, 'CW'): 'CD2', ('HIE', 14, 'H4'): 'HD2', ('HIE', 15, 'C'): 'C', ('HIE', 16, 'O'): 'O', ('GLN', 16, 'O2'): 'O', ('GLN', 17, 'O2'): 'OXT', ('MET', 0, 'N3'): 'N', ('MET', 2, 'H'): 'H2', ('MET', 3, 'H'): 'H3', ('MET', 4, 'CX'): 'CA', ('MET', 5, 'HP'): 'HA', ('MET', 6, '2C'): 'CB', ('MET', 7, 'HC'): 'HB2', ('MET', 8, 'HC'): 'HB3', ('MET', 9, '2C'): 'CG', ('MET', 10, 'H1'): 'HG2', ('MET', 11, 'H1'): 'HG3', ('MET', 12, 'S'): 'SD', ('MET', 13, 'CT'): 'CE', ('MET', 15, 'H1'): 'HE2', ('MET', 16, 'H1'): 'HE3', ('MET', 17, 'C'): 'C', ('MET', 18, 'O'): 'O', ('GLU', 0, 'N3'): 'N', ('GLU', 2, 'H'): 'H2', ('GLU', 3, 'H'): 'H3', ('GLU', 4, 'CX'): 'CA', ('GLU', 5, 'HP'): 'HA', ('GLU', 6, '2C'): 'CB', ('GLU', 7, 'HC'): 'HB2', ('GLU', 9, '2C'): 'CG', ('GLU', 10, 'HC'): 'HG2', ('GLU', 11, 'HC'): 'HG3', ('GLU', 12, 'CO'): 'CD', ('GLU', 13, 'O2'): 'OE1', ('GLU', 14, 'O2'): 'O', ('GLU', 15, 'C'): 'C', ('GLU', 16, 'O'): 'O', ('LYS', 21, 'O2'): 'O', ('LYS', 22, 'O2'): 'OXT', ('SER', 0, 'N3'): 'N', ('SER', 2, 'H'): 'H2', ('SER', 3, 'H'): 'H3', ('SER', 4, 'CX'): 'CA', ('SER', 5, 'HP'): 'HA', ('SER', 6, '2C'): 'CB', ('SER', 7, 'H1'): 'HB2', ('SER', 8, 'H1'): 'HB3', ('SER', 9, 'OH'): 'OG', ('SER', 10, 'HO'): 'HG', ('SER', 11, 'C'): 'C', ('SER', 12, 'O'): 'O', ('PRO', 13, 'O2'): 'O', ('PRO', 14, 'O2'): 'OXT', ('ILE', 0, 'N3'): 'N', ('ILE', 2, 'H'): 'H2', ('ILE', 3, 'H'): 'H3', ('ILE', 4, 'CX'): 'CA', ('ILE', 5, 'HP'): 'HA', ('ILE', 6, '3C'): 'CB', ('ILE', 8, 'CT'): 'CG2', ('ILE', 10, 'HC'): 'HG22', ('ILE', 12, '2C'): 'CG1', ('ILE', 13, 'HC'): 'HG12', ('ILE', 15, 'CT'): 'CD1', ('ILE', 17, 'HC'): 'HD12', ('ILE', 18, 'HC'): 'HD13', ('ILE', 19, 'C'): 'C', ('ILE', 20, 'O'): 'O', ('ASP', 0, 'N3'): 'N', ('ASP', 2, 'H'): 'H2', ('ASP', 3, 'H'): 'H3', ('ASP', 4, 'CX'): 'CA', ('ASP', 5, 'HP'): 'HA', ('ASP', 6, '2C'): 'CB', ('ASP', 7, 'HC'): 'HB2', ('ASP', 8, 'HC'): 'HB3', ('ASP', 9, 'CO'): 'CG', ('ASP', 10, 'O2'): 'OD1', ('ASP', 11, 'O2'): 'O', ('ASP', 12, 'C'): 'C', ('ASP', 13, 'O'): 'O', ('ALA', 0, 'N3'): 'N', ('ALA', 2, 'H'): 'H2', ('ALA', 3, 'H'): 'H3', ('ALA', 4, 'CX'): 'CA', ('ALA', 5, 'HP'): 'HA', ('ALA', 6, 'CT'): 'CB', ('ALA', 8, 'HC'): 'HB2', ('ALA', 9, 'HC'): 'HB3', ('ALA', 10, 'C'): 'C', ('ALA', 11, 'O'): 'O', ('CYX', 0, 'N'): 'N', ('CYX', 1, 'H'): 'H', ('CYX', 2, 'CX'): 'CA', ('CYX', 3, 'H1'): 'HA', ('CYX', 4, '2C'): 'CB', ('CYX', 5, 'H1'): 'HB2', ('CYX', 6, 'H1'): 'HB3', ('CYX', 7, 'S'): 'SG', ('CYX', 8, 'C'): 'C', ('CYX', 9, 'O'): 'O', ('GLU', 15, 'O2'): 'OXT', ('VAL', 0, 'N3'): 'N', ('VAL', 2, 'H'): 'H2', ('VAL', 3, 'H'): 'H3', ('VAL', 4, 'CX'): 'CA', ('VAL', 5, 'HP'): 'HA', ('VAL', 6, '3C'): 'CB', ('VAL', 8, 'CT'): 'CG1', ('VAL', 10, 'HC'): 'HG12', ('VAL', 12, 'CT'): 'CG2', ('VAL', 14, 'HC'): 'HG22', ('VAL', 15, 'HC'): 'HG23', ('VAL', 16, 'C'): 'C', ('VAL', 17, 'O'): 'O', ('TYR', 20, 'O2'): 'O', ('TYR', 21, 'O2'): 'OXT', ('GLN', 0, 'N3'): 'N', ('GLN', 2, 'H'): 'H2', ('GLN', 3, 'H'): 'H3', ('GLN', 4, 'CX'): 'CA', ('GLN', 5, 'HP'): 'HA', ('GLN', 6, '2C'): 'CB', ('GLN', 7, 'HC'): 'HB2', ('GLN', 9, '2C'): 'CG', ('GLN', 10, 'HC'): 'HG2', ('GLN', 11, 'HC'): 'HG3', ('GLN', 12, 'C'): 'CD', ('GLN', 13, 'O'): 'OE1', ('GLN', 14, 'N'): 'NE2', ('GLN', 15, 'H'): 'HE21', ('GLN', 16, 'H'): 'HE22', ('GLN', 17, 'C'): 'C', ('GLN', 18, 'O'): 'O', ('THR', 0, 'N3'): 'N', ('THR', 2, 'H'): 'H2', ('THR', 3, 'H'): 'H3', ('THR', 4, 'CX'): 'CA', ('THR', 5, 'HP'): 'HA', ('THR', 6, '3C'): 'CB', ('THR', 7, 'H1'): 'HB', ('THR', 8, 'CT'): 'CG2', ('THR', 10, 'HC'): 'HG22', ('THR', 11, 'HC'): 'HG23', ('THR', 12, 'OH'): 'OG1', ('THR', 13, 'HO'): 'HG1', ('THR', 14, 'C'): 'C', ('THR', 15, 'O'): 'O', ('ARG', 23, 'O2'): 'O', ('ARG', 24, 'O2'): 'OXT', ('THR', 13, 'O2'): 'O', ('THR', 14, 'O2'): 'OXT', ('GLY', 0, 'N3'): 'N', ('GLY', 2, 'H'): 'H2', ('GLY', 3, 'H'): 'H3', ('GLY', 4, 'CX'): 'CA', ('GLY', 5, 'HP'): 'HA2', ('GLY', 6, 'HP'): 'HA3', ('GLY', 7, 'C'): 'C', ('GLY', 8, 'O'): 'O', ('PHE', 19, 'O2'): 'O', ('PHE', 20, 'O2'): 'OXT', ('LEU', 18, 'O2'): 'O', ('LEU', 19, 'O2'): 'OXT', ('ILE', 18, 'O2'): 'O', ('ILE', 19, 'O2'): 'OXT', ('LYS', 0, 'N3'): 'N', ('LYS', 2, 'H'): 'H2', ('LYS', 3, 'H'): 'H3', ('LYS', 4, 'CX'): 'CA', ('LYS', 5, 'HP'): 'HA', ('LYS', 6, 'C8'): 'CB', ('LYS', 7, 'HC'): 'HB2', ('LYS', 9, 'C8'): 'CG', ('LYS', 10, 'HC'): 'HG2', ('LYS', 12, 'C8'): 'CD', ('LYS', 13, 'HC'): 'HD2', ('LYS', 14, 'HC'): 'HD3', ('LYS', 15, 'C8'): 'CE', ('LYS', 16, 'HP'): 'HE2', ('LYS', 17, 'HP'): 'HE3', ('LYS', 18, 'N3'): 'NZ', ('LYS', 20, 'H'): 'HZ2', ('LYS', 21, 'H'): 'HZ3', ('LYS', 22, 'C'): 'C', ('LYS', 23, 'O'): 'O', ('HIE', 0, 'N3'): 'N', ('HIE', 2, 'H'): 'H2', ('HIE', 3, 'H'): 'H3', ('HIE', 4, 'CX'): 'CA', ('HIE', 5, 'HP'): 'HA', ('HIE', 6, 'CT'): 'CB', ('HIE', 7, 'HC'): 'HB2', ('HIE', 8, 'HC'): 'HB3', ('HIE', 9, 'CC'): 'CG', ('HIE', 10, 'NB'): 'ND1', ('HIE', 11, 'CR'): 'CE1', ('HIE', 12, 'H5'): 'HE1', ('HIE', 13, 'NA'): 'NE2', ('HIE', 14, 'H'): 'HE2', ('HIE', 15, 'CW'): 'CD2', ('HIE', 16, 'H4'): 'HD2', ('HIE', 17, 'C'): 'C', ('HIE', 18, 'O'): 'O', ('GLY', 6, 'O2'): 'O', ('GLY', 7, 'O2'): 'OXT', ('ASN', 13, 'O2'): 'O', ('ASN', 14, 'O2'): 'OXT', ('LEU', 0, 'N3'): 'N', ('LEU', 2, 'H'): 'H2', ('LEU', 3, 'H'): 'H3', ('LEU', 4, 'CX'): 'CA', ('LEU', 5, 'HP'): 'HA', ('LEU', 6, '2C'): 'CB', ('LEU', 7, 'HC'): 'HB2', ('LEU', 9, '3C'): 'CG', ('LEU', 11, 'CT'): 'CD1', ('LEU', 13, 'HC'): 'HD12', ('LEU', 15, 'CT'): 'CD2', ('LEU', 17, 'HC'): 'HD22', ('LEU', 18, 'HC'): 'HD23', ('LEU', 19, 'C'): 'C', ('LEU', 20, 'O'): 'O', ('SER', 10, 'O2'): 'O', ('SER', 11, 'O2'): 'OXT', ('ASN', 0, 'N3'): 'N', ('ASN', 2, 'H'): 'H2', ('ASN', 3, 'H'): 'H3', ('ASN', 4, 'CX'): 'CA', ('ASN', 5, 'HP'): 'HA', ('ASN', 6, '2C'): 'CB', ('ASN', 7, 'HC'): 'HB2', ('ASN', 8, 'HC'): 'HB3', ('ASN', 9, 'C'): 'CG', ('ASN', 10, 'O'): 'OD1', ('ASN', 11, 'N'): 'ND2', ('ASN', 12, 'H'): 'HD21', ('ASN', 13, 'H'): 'HD22', ('ASN', 14, 'C'): 'C', ('ASN', 15, 'O'): 'O', ('ARG', 0, 'N3'): 'N', ('ARG', 2, 'H'): 'H2', ('ARG', 3, 'H'): 'H3', ('ARG', 4, 'CX'): 'CA', ('ARG', 5, 'HP'): 'HA', ('ARG', 6, 'C8'): 'CB', ('ARG', 7, 'HC'): 'HB2', ('ARG', 9, 'C8'): 'CG', ('ARG', 10, 'HC'): 'HG2', ('ARG', 11, 'HC'): 'HG3', ('ARG', 12, 'C8'): 'CD', ('ARG', 13, 'H1'): 'HD2', ('ARG', 14, 'H1'): 'HD3', ('ARG', 15, 'N2'): 'NE', ('ARG', 16, 'H'): 'HE', ('ARG', 17, 'CA'): 'CZ', ('ARG', 18, 'N2'): 'NH1', ('ARG', 19, 'H'): 'HH11', ('ARG', 21, 'N2'): 'NH2', ('ARG', 22, 'H'): 'HH21', ('ARG', 23, 'H'): 'HH22', ('ARG', 24, 'C'): 'C', ('ARG', 25, 'O'): 'O', ('MET', 16, 'O2'): 'O', ('MET', 17, 'O2'): 'OXT', ('PHE', 0, 'N3'): 'N', ('PHE', 2, 'H'): 'H2', ('PHE', 3, 'H'): 'H3', ('PHE', 4, 'CX'): 'CA', ('PHE', 5, 'HP'): 'HA', ('PHE', 6, 'CT'): 'CB', ('PHE', 7, 'HC'): 'HB2', ('PHE', 8, 'HC'): 'HB3', ('PHE', 9, 'CA'): 'CG', ('PHE', 18, 'CA'): 'CD2', ('PHE', 19, 'HA'): 'HD2', ('PHE', 20, 'C'): 'C', ('PHE', 21, 'O'): 'O', ('ASP', 12, 'O2'): 'OXT', ('CYS', 0, 'N3'): 'N', ('CYS', 2, 'H'): 'H2', ('CYS', 3, 'H'): 'H3', ('CYS', 4, 'CX'): 'CA', ('CYS', 5, 'HP'): 'HA', ('CYS', 6, '2C'): 'CB', ('CYS', 7, 'H1'): 'HB2', ('CYS', 8, 'H1'): 'HB3', ('CYS', 9, 'SH'): 'SG', ('CYS', 10, 'HS'): 'HG', ('CYS', 11, 'C'): 'C', ('CYS', 12, 'O'): 'O', ('ALA', 9, 'O2'): 'O', ('ALA', 10, 'O2'): 'OXT', ('TRP', 23, 'O2'): 'O', ('TRP', 24, 'O2'): 'OXT', ('HIE', 16, 'O2'): 'O', ('HIE', 17, 'O2'): 'OXT', ('CYX', 9, 'O2'): 'O', ('CYX', 10, 'O2'): 'OXT', ('TYR', 0, 'N3'): 'N', ('TYR', 2, 'H'): 'H2', ('TYR', 3, 'H'): 'H3', ('TYR', 4, 'CX'): 'CA', ('TYR', 5, 'HP'): 'HA', ('TYR', 6, 'CT'): 'CB', ('TYR', 7, 'HC'): 'HB2', ('TYR', 8, 'HC'): 'HB3', ('TYR', 9, 'CA'): 'CG', ('TYR', 12, 'CA'): 'CE1', ('TYR', 13, 'HA'): 'HE1', ('TYR', 14, 'C'): 'CZ', ('TYR', 15, 'OH'): 'OH', ('TYR', 16, 'HO'): 'HH', ('TYR', 19, 'CA'): 'CD2', ('TYR', 20, 'HA'): 'HD2', ('TYR', 21, 'C'): 'C', ('TYR', 22, 'O'): 'O', ('VAL', 15, 'O2'): 'O', ('VAL', 16, 'O2'): 'OXT', ('CYX', 0, 'N3'): 'N', ('CYX', 2, 'H'): 'H2', ('CYX', 3, 'H'): 'H3', ('CYX', 4, 'CX'): 'CA', ('CYX', 5, 'HP'): 'HA', ('CYX', 6, '2C'): 'CB', ('CYX', 7, 'H1'): 'HB2', ('CYX', 8, 'H1'): 'HB3', ('CYX', 9, 'S'): 'SG', ('CYX', 10, 'C'): 'C', ('CYX', 11, 'O'): 'O', ('CYS', 10, 'O2'): 'O', ('CYS', 11, 'O2'): 'OXT', ('TRP', 0, 'N3'): 'N', ('TRP', 2, 'H'): 'H2', ('TRP', 3, 'H'): 'H3', ('TRP', 4, 'CX'): 'CA', ('TRP', 5, 'HP'): 'HA', ('TRP', 6, 'CT'): 'CB', ('TRP', 7, 'HC'): 'HB2', ('TRP', 8, 'HC'): 'HB3', ('TRP', 9, 'C*'): 'CG', ('TRP', 10, 'CW'): 'CD1', ('TRP', 11, 'H4'): 'HD1', ('TRP', 12, 'NA'): 'NE1', ('TRP', 13, 'H'): 'HE1', ('TRP', 14, 'CN'): 'CE2', ('TRP', 21, 'CA'): 'CE3', ('TRP', 22, 'HA'): 'HE3', ('TRP', 23, 'CB'): 'CD2', ('TRP', 24, 'C'): 'C', ('TRP', 25, 'O'): 'O'}

    atom_index2name_dict {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 34: 'Se', 35: 'Br', 53: 'I'}
    """
    # get standard atom name for proteins
    # CA, C, N, H, H2, H3, HA, CB, O ...
    protein_atom_index, sequences = get_atom_and_residue_type(protein_atom_index, residue_index, atom_index, protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, molecules_begin_atom_index)
    mask_backbone = (protein_atom_index == "CA") | (protein_atom_index == "C") | (protein_atom_index == "N")

    protein_backbone_atom_type = protein_atom_index[mask_backbone]
    protein_mask_ca = protein_backbone_atom_type == "CA"
    protein_mask_c = protein_backbone_atom_type == "C"
    protein_mask_n = protein_backbone_atom_type == "N"
    assert np.sum(protein_mask_ca) == np.sum(protein_mask_c) == np.sum(protein_mask_n)
    return mask_backbone, protein_mask_ca, protein_mask_c, protein_mask_n, sequences


def parse_MISATO_data(misato_data, atom_index2name_dict, atom_num2atom_mass, residue_index2name_dict, protein_atom_index2standard_name_dict, atom_reisdue2standard_atom_name_dict):
    ligand_begin_index = misato_data["molecules_begin_atom_index"][:][-1]

    atom_index = misato_data["atoms_number"][:]
    protein_atom_index = misato_data["atoms_type"][:]
    residue_number = misato_data["atoms_residue"][:]

    frames_interaction_energy = np.expand_dims(misato_data["frames_interaction_energy"][0], 0) # 1

    # for protein
    protein_coordinates = misato_data["trajectory_coordinates"][0][:][:ligand_begin_index]
    protein_mask_backbone, protein_mask_ca, protein_mask_c, protein_mask_n, protein_sequence = extract_sequence_and_structure(protein_atom_index[:ligand_begin_index], residue_number[:ligand_begin_index], atom_index[:ligand_begin_index], protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, misato_data["molecules_begin_atom_index"][:][:-1])
    protein_coordinates = protein_coordinates[protein_mask_backbone, :]
    protein_coordinates = torch.tensor(protein_coordinates, dtype=torch.float32)
    protein_residue = residue_number[:ligand_begin_index][protein_mask_backbone][protein_mask_ca]
    protein_residue = torch.tensor(protein_residue, dtype=torch.int64)
    assert protein_residue.min() >= 1
    protein_residue -= 1
    protein_mask_ca = torch.tensor(protein_mask_ca, dtype=torch.bool)
    protein_mask_c = torch.tensor(protein_mask_c, dtype=torch.bool)
    protein_mask_n = torch.tensor(protein_mask_n, dtype=torch.bool)

    # for peptide
    peptide_coordinates = misato_data["trajectory_coordinates"][0][:][ligand_begin_index:]
    peptide_mask_backbone, peptide_mask_ca, peptide_mask_c, peptide_mask_n, peptide_sequence = extract_sequence_and_structure(protein_atom_index[ligand_begin_index:], residue_number[ligand_begin_index:], atom_index[ligand_begin_index:], protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, [misato_data["molecules_begin_atom_index"][:][-1]])
    peptide_coordinates = peptide_coordinates[peptide_mask_backbone, :]
    peptide_coordinates = torch.tensor(peptide_coordinates, dtype=torch.float32)
    peptide_residue = residue_number[ligand_begin_index:][peptide_mask_backbone][peptide_mask_ca]
    peptide_residue = torch.tensor(peptide_residue, dtype=torch.int64)
    assert peptide_residue.min() >= 1
    peptide_residue -= 1
    peptide_mask_ca = torch.tensor(peptide_mask_ca, dtype=torch.bool)
    peptide_mask_c = torch.tensor(peptide_mask_c, dtype=torch.bool)
    peptide_mask_n = torch.tensor(peptide_mask_n, dtype=torch.bool)

    # 1
    frames_interaction_energy = torch.tensor(frames_interaction_energy, dtype=torch.float32)

    data = Data(
        protein_pos=protein_coordinates,
        protein_residue=protein_residue,
        protein_mask_ca=protein_mask_ca,
        protein_mask_c=protein_mask_c,
        protein_mask_n=protein_mask_n,
        # 
        peptide_pos=peptide_coordinates,
        peptide_residue=peptide_residue,
        peptide_mask_ca=peptide_mask_ca,
        peptide_mask_c=peptide_mask_c,
        peptide_mask_n=peptide_mask_n,
        # 
        energy=frames_interaction_energy,
    )
    return data, protein_sequence, peptide_sequence


class MISATODataset(InMemoryDataset):
    def __init__(self, root):
        self.root = root
        self.c_alpha_atom_type = 6
        self.c_atom_type = 3
        self.n_atom_type = 24
        super(MISATODataset, self).__init__(root, None, None, None)

        self.data, self.slices = torch.load(self.processed_paths[0])

        f_ = open(self.processed_paths[1], "r")
        PDB_idx_list, peptide_sequence_list, protein_sequence_list = [], [], []
        for line in f_.readlines():
            line = line.strip()
            line = line.split(",")
            peptide_idx = line[0]
            peptide_sequence = line[1]
            protein_sequence = line[2]
            PDB_idx_list.append(peptide_idx)
            peptide_sequence_list.append(peptide_sequence)
            protein_sequence_list.append(protein_sequence)
        self.PDB_idx_list = PDB_idx_list
        self.peptide_sequence_list = peptide_sequence_list
        self.protein_sequence_list = protein_sequence_list
        return

    @property
    def raw_file_names(self):
        file_name = "MD.hdf5"
        return [file_name]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "PDB_id_and_sequence.txt"]

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    def process(self):
        MD_file_path = self.raw_paths[0]
        MD_data = h5py.File(MD_file_path, "r")

        residue_index2name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_residue_map.pickle'),'rb'))
        protein_atom_index2standard_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_type_map.pickle'),'rb'))
        atom_reisdue2standard_atom_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_name_map_for_pdb.pickle'),'rb'))

        peptides_file = os.path.join(utils_dir, "peptides.txt")
        peptides_idx_list = []
        with open(peptides_file) as f:
            for line in f.readlines():
                peptides_idx_list.append(line.strip().upper())

        atom_index2name_dict = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 20:'Ca', 34:'Se', 35:'Br', 53:'I'}

        import importlib.resources
        import ProteinDT.datasets
        import pandas as pd
        with importlib.resources.path(ProteinDT.datasets, 'periodic_table.csv') as file_name:
            periodic_table_file = file_name
        periodic_table_data = pd.read_csv(periodic_table_file)
        atom_num2atom_mass = {}
        for i in range(1, 119):
            atom_mass = periodic_table_data.loc[i-1]['AtomicMass']
            atom_num2atom_mass[i] = atom_mass

        data_list, PDB_idx_list, protein_sequence_list, peptide_sequence_list = [], [], [], []
        for idx in tqdm(peptides_idx_list):
            try:
                misato_data = MD_data[idx]
                print(idx)
            except:
                continue

            if misato_data is None:
                print("misato_data", misato_data, idx)
                continue
            data,  protein_sequence, peptide_sequence = parse_MISATO_data(
                misato_data, atom_index2name_dict=atom_index2name_dict, atom_num2atom_mass=atom_num2atom_mass,
                residue_index2name_dict=residue_index2name_dict, protein_atom_index2standard_name_dict=protein_atom_index2standard_name_dict, atom_reisdue2standard_atom_name_dict=atom_reisdue2standard_atom_name_dict)
            protein_sequence = ''.join(protein_sequence)
            peptide_sequence = ''.join(peptide_sequence)
            if not (5 <= len(peptide_sequence) <= 20):
                continue
            data_list.append(data)
            PDB_idx_list.append(idx)
            protein_sequence_list.append(protein_sequence)
            peptide_sequence_list.append(peptide_sequence)
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        f_ = open(self.processed_paths[1], "w")
        for PDB_idx, peptide_sequence, protein_sequence in zip(PDB_idx_list, peptide_sequence_list, protein_sequence_list):
            print("{},{},{}".format(PDB_idx, peptide_sequence, protein_sequence), file=f_)
        f_.flush()
        f_.close()
        
        return

    def get_peptide_idx2data(self):
        record = {}
        for i in range(len(self.PDB_idx_list)):
            PDB_idx = self.PDB_idx_list[i]
            data = self.get(i)
            record[PDB_idx] = data
        return record


if __name__ == "__main__":    
    data_root_list = ["../../../data/MISATO"]
    for data_root in data_root_list:
        dataset = MISATODataset(data_root)
