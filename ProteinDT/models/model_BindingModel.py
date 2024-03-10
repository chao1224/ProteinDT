import torch
import torch.nn as nn
from .model_CDConv import CD_Convolution


class BindingModel(nn.Module):
    def __init__(self, args):
        super(BindingModel, self).__init__()

        num_class = 1

        geometric_radii = [x * args.CDConv_radius for x in args.CDConv_geometric_raddi_coeff]

        self.peptide_model = CD_Convolution(
            geometric_radii=geometric_radii,
            sequential_kernel_size=args.CDConv_kernel_size,
            kernel_channels=args.CDConv_kernel_channels, channels=args.CDConv_channels, base_width=args.CDConv_base_width,
            num_classes=num_class)

        self.protein_model = CD_Convolution(
            geometric_radii=geometric_radii,
            sequential_kernel_size=args.CDConv_kernel_size,
            kernel_channels=args.CDConv_kernel_channels, channels=args.CDConv_channels, base_width=args.CDConv_base_width,
            num_classes=num_class)

        return

    def forward(
        self, 
        protein_residue, protein_pos_N, protein_pos_Ca, protein_pos_C, protein_batch,
        peptide_residue, peptide_pos_N, peptide_pos_Ca, peptide_pos_C, peptide_batch,
    ):

        protein_residue_repr = self.protein_model(
            pos=protein_pos_Ca,
            seq=protein_residue,
            batch=protein_batch,
        )

        peptide_residue_repr = self.peptide_model(
            pos=peptide_pos_Ca,
            seq=peptide_residue,
            batch=peptide_batch,
        )
        binding_energy = protein_residue_repr + peptide_residue_repr
        
        return binding_energy
