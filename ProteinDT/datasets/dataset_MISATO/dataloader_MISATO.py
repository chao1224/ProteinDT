import random
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


class BatchMISATO(Data):
    """A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier."""

    def __init__(self, batch=None, **kwargs):
        super(BatchMISATO, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMISATO()

        for key in keys:
            batch[key] = []
        batch.peptide_batch = []
        batch.protein_batch = []

        cumsum_node_peptide = 0
        cumsum_node_protein = 0

        for i, data in enumerate(data_list):
            num_nodes_peptide = data.peptide_residue.size()[0]
            num_nodes_protein = data.protein_residue.size()[0]

            batch.peptide_batch.append(torch.full((num_nodes_peptide,), i, dtype=torch.long))
            batch.protein_batch.append(torch.full((num_nodes_protein,), i, dtype=torch.long))

            for key in data.keys:
                item = data[key]
                batch[key].append(item)

            cumsum_node_peptide += num_nodes_peptide
            cumsum_node_protein += num_nodes_protein

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.peptide_batch = torch.cat(batch.peptide_batch, dim=-1)
        batch.protein_batch = torch.cat(batch.protein_batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class MISATODataLoader(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(MISATODataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMISATO.from_data_list(data_list),
            **kwargs)