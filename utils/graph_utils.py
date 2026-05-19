PROPERTY_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "QED",
]


def smiles_to_graph(smiles):
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = [[float(atom.GetAtomicNum())] for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges).t().contiguous()

    return Data(x=x, edge_index=edge_index)