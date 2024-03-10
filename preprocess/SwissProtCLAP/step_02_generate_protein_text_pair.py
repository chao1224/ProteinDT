import os
import re


if __name__ == "__main__":
    base_folder = "./data"
    sequence_file = os.path.join(base_folder, "uniprot_sprot_xml_seq.txt")
    func_file = os.path.join(base_folder, "uniprot_sprot_xml_func.txt")
    text_file = os.path.join(base_folder, "uniprot_sprot_xml_text.txt")

    uniprot_id2seq = {}
    f = open(sequence_file, 'r')
    for line in f.readlines():
        line = line.strip().split(",")
        uniprot, sequence = line
        uniprot_id2seq[uniprot] = sequence
    print("len of seq {}".format(len(uniprot_id2seq)))
    
    uniprot_id2func = {}
    f = open(func_file, 'r')
    for line_idx, line in enumerate(f.readlines()):
        line = line.strip()
        if line_idx % 2 == 0:
            uniprot = line
        else:
            func = line
            uniprot_id2func[uniprot] = func
    print("len of func {}".format(len(uniprot_id2func)))
    
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

    ########## output ##########
    output_folder = "../../data"
    output_file = os.path.join(output_folder, "SwissProtCLAP", "text_sequence.txt")
    f = open(output_file, 'w')
    for uniprot_id, text in uniprot_id2text.items():
        assert uniprot_id in uniprot_id2seq
        print(uniprot_id, file=f)
        print(text, file=f)

    output_file = os.path.join(output_folder, "SwissProtCLAP", "protein_sequence.txt")
    f = open(output_file, 'w')
    for uniprot_id, _ in uniprot_id2text.items():
        sequence = uniprot_id2seq[uniprot_id]
        sequence = re.sub(r"[UZOB]", "X", sequence)
        print(uniprot_id, file=f)
        print(sequence, file=f)
