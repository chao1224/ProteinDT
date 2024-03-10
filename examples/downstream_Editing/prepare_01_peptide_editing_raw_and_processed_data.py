import os
import random
random.seed(42)
from ProteinDT.datasets import MISATODataset
import requests
import json


text_file = "datasets_and_checkpoints/uniprot_sprot_xml_text.txt"
uniprot_id2text = {}
f = open(text_file, 'r')
for line_idx, line in enumerate(f.readlines()):
    line = line.strip()
    if line_idx % 2 == 0:
        uniprot = line
    else:
        text = line
        uniprot_id2text[uniprot] = text
print("len of text {}".format(len(uniprot_id2text)))


if __name__ == "__main__":
    data_folder = "../../data/MISATO"
    dataset = MISATODataset(data_folder)

    PDB_idx_list = dataset.PDB_idx_list
    peptide_sequence_list = dataset.peptide_sequence_list
    protein_sequence_list = dataset.protein_sequence_list
    print("peptide_sequence_list", len(peptide_sequence_list))
    print("PDB_idx_list", len(PDB_idx_list))
    print("protein_sequence_list", len(protein_sequence_list))

    output_dir = "datasets_and_checkpoints/peptide_binding/MISATO"
    os.makedirs(output_dir, exist_ok = True)

    idx_list = [x for x in range(len(PDB_idx_list))]
    print("idx_list", idx_list)
    random.shuffle(idx_list)
    print("idx_list", idx_list)

    PDB_idx2text = {}
    for PDB_idx in PDB_idx_list:
        description_url = "https://data.rcsb.org/rest/v1/core/uniprot/{entry_id}/1".format(entry_id=PDB_idx)
        description_data = requests.get(description_url).json()
        
        try:
            description_data = description_data[0]
            uniprot_idx = description_data["rcsb_uniprot_accession"][0]
            PDB_idx2text[PDB_idx] = uniprot_id2text[uniprot_idx]
        except:
            print("error with PDB: {}".format(PDB_idx))
            continue
    print("PDB_idx2text: {}".format(len(PDB_idx2text)))

    f = open("{}/PDB_idx2text.json".format(output_dir), "w")
    json.dump(PDB_idx2text, f)

    f = open("{}/PDB_mapping_data.txt".format(output_dir), "w")
    for i in idx_list:
        PDB_idx, peptide_sequence, protein_sequence = PDB_idx_list[i], peptide_sequence_list[i], protein_sequence_list[i]
        if PDB_idx not in PDB_idx2text:
            continue
        print("{},{},{}".format(PDB_idx, peptide_sequence, protein_sequence), file=f)
    f.flush()
    f.close()

    f = open("{}/preprocessed_data.csv".format(output_dir), "w")
    for i in idx_list:
        peptide_sequence = peptide_sequence_list[i]
        PDB_idx, peptide_sequence, protein_sequence = PDB_idx_list[i], peptide_sequence_list[i], protein_sequence_list[i]
        if PDB_idx not in PDB_idx2text:
            continue
        print("{}".format(peptide_sequence), file=f)
    f.flush()
    f.close()