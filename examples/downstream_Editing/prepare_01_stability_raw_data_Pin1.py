import os
import json
from random import shuffle


if __name__ == "__main__":
    input_json_file = "datasets_and_checkpoints/stability/stability_test.json"
    f = open(input_json_file, 'r')
    data = json.load(f)

    record_list = []
    for idx, record in enumerate(data):
        if 'Pin1' not in record['topology']:
            continue
        protein_sequence = record["primary"]
        assert protein_sequence[:1] == "G"
        assert protein_sequence[-3:] == "GSS"
        protein_sequence = protein_sequence[1:-3]
        assert len(protein_sequence) == 39
        value = record["stability_score"][0]
        record_list.append([protein_sequence, value])
    print("len of valid record", len(record_list))

    shuffle(record_list)
    N = len(record_list)
    
    split_size = [0.8, 0.1, 0.1]
    train_size = int(split_size[0] * N)
    train_val_size = int((split_size[0] + split_size[1]) * N)

    output_dir = "datasets_and_checkpoints/stability/Pin1"
    os.makedirs(output_dir, exist_ok = True)

    f = open("{}/train_data.txt".format(output_dir), "w")
    for idx in range(0, train_size):
        protein_sequence, label = record_list[idx]
        print("{},{}".format(protein_sequence, label), file=f)
    f.flush()
    f.close()

    f = open("{}/val_data.txt".format(output_dir), "w")
    for idx in range(train_size, train_val_size):
        protein_sequence, label = record_list[idx]
        print("{},{}".format(protein_sequence, label), file=f)
    f.flush()
    f.close()

    f = open("{}/test_data.txt".format(output_dir), "w")
    for idx in range(train_val_size, N):
        protein_sequence, label = record_list[idx]
        print("{},{}".format(protein_sequence, label), file=f)
    f.flush()
    f.close()
