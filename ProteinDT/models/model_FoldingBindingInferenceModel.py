import torch
import torch.nn as nn
from .model_BindingModel import BindingModel
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdb = to_pdb(pred)
        pdbs.append(pdb)
    return pdbs


class FoldingBindingInferenceModel(nn.Module):
    def __init__(self, input_model_path=None):
                
        super(FoldingBindingInferenceModel, self).__init__()

        self.folding_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.folding_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

        config = {
            "CDConv_radius": 4,
            "CDConv_geometric_raddi_coeff": [2, 3, 4, 5],
            "CDConv_kernel_size": 21,
            "CDConv_kernel_channels": [24],
            "CDConv_channels": [256, 512, 1024, 2048],
            "CDConv_base_width": 64,
        }
        config = AttrDict(config)

        self.binding_model = BindingModel(config)
        if input_model_path is not None:
            from collections import OrderedDict
            print("Loading protein model from {}...".format(input_model_path))
            state_dict = torch.load(input_model_path, map_location='cpu')["binding_model"]
            self.binding_model.load_state_dict(state_dict)

        return

    def folding(self, protein_sequence):
        tokenized_input = self.folding_tokenizer(protein_sequence, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.cuda()

        with torch.no_grad():
            output = self.folding_model(tokenized_input)

        pdb_list = convert_outputs_to_pdb(output)
        return pdb_list

    def binding(
        self,
        protein_residue, protein_pos_N, protein_pos_Ca, protein_pos_C, protein_batch,
        peptide_residue, peptide_pos_N, peptide_pos_Ca, peptide_pos_C, peptide_batch,
    ):
        energy = self.binding_model(
            protein_residue, protein_pos_N, protein_pos_Ca, protein_pos_C, protein_batch,
            peptide_residue, peptide_pos_N, peptide_pos_Ca, peptide_pos_C, peptide_batch,
        )
        return energy
