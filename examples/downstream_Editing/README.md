# Datasets and Checkpoints Preparation

## 1 Structure Editing

### 1.1 Dataset
This follows the [ChatDrug](https://github.com/chao1224/ChatDrug).

```
mkdir -p datasets_and_checkpoints/structure
cp -r ../../data/downstream_datasets/secondary_structure/*.lmdb datasets_and_checkpoints/structure/

python prepare_02_processed_data.py --editing_task=alpha
```

### 1.2 Oracle Evaluator
We can use this checkpoint:
```
mkdir -p datasets_and_checkpoints/structure/oracle/
cp ../../output/ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5/downstream_TAPE/ss3/3-3e-5-5-2-8-0.08/pytorch_model.bin \
    datasets_and_checkpoints/structure/oracle/pytorch_model_ss3.bin
```

Or use the following huggingface link:
```
from huggingface_hub import hf_hub_download

hf_hub_download(
  repo_id="chao1224/ProteinCLAP_pretrain_EBM_NCE_downstream_property_prediction",
  repo_type="model",
  filename="pytorch_model_ss3.bin",
  local_dir="datasets_and_checkpoints/structure/oracle")
```

## 2 Stability Editing

### 2.1 Dataset
```
mkdir -p datasets_and_checkpoints/stability
cp ../../data/downstream_datasets/stability/*.json    datasets_and_checkpoints/stability/

python prepare_01_stability_raw_data_Villin.py
python prepare_01_stability_raw_data_Pin1.py

python prepare_02_processed_data.py --editing_task=Villin
python prepare_02_processed_data.py --editing_task=Pin1
```

### 2.2 Oracle Evaluator
We can use this checkpoint:
```
mkdir -p datasets_and_checkpoints/stability/oracle/

cp ../../output/downstream_TAPE/stability/ProtBERT_BFD/3-3e-5-5-2-16-0.08/pytorch_model.bin \
    datasets_and_checkpoints/stability/oracle/pytorch_model_stability.bin
```

## 3 Peptide Editing

### 3.1 Dataset

Refer to this [GitHub link](https://github.com/t7morgen/misato-dataset).

```
cd .../../data

mkdir -r MISATO/raw
cd MISATO/raw

wget https://zenodo.org/record/7711953/files/MD.hdf5?download=1
mv 'MD.hdf5?download=1' MD.hdf5
wget https://zenodo.org/record/7711953/files/train_MD.txt?download=1
mv 'train_MD.txt?download=1' train_MD.txt
wget https://zenodo.org/record/7711953/files/val_MD.txt?download=1
mv 'val_MD.txt?download=1' val_MD.txt
wget https://zenodo.org/record/7711953/files/test_MD.txt?download=1
mv 'test_MD.txt?download=1' test_MD.txt
```

Back to this folder, and do the following:
```
python prepare_01_peptide_editing_raw_and_processed_data.py
```

### 3.2 Oracle Evaluator

We will have to train an oracle model by ourselves + docking.

```
python prepare_03_train_peptide_editing_evaluator.py --epochs=1000 --lr=1e-4 --output_model_dir=datasets_and_checkpoints/peptide_binding/MISATO
```
