import os
from omegaconf import OmegaConf, open_dict

import torch
import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
from rdkit import Chem
import re

def create_folders(args):
    try:
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass

def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch, max_num_nodes=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model
    saved_dataset = saved_cfg.dataset
    
    for key, val in saved_dataset.items():
        OmegaConf.set_struct(cfg.dataset, True)
        with open_dict(cfg.dataset):
            if key not in cfg.dataset.keys():
                setattr(cfg.dataset, key, val)

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X 
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor, categorical: bool = False):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if categorical:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

class ValidityChecker:
    def __init__(self, sample, atom_decoder):
        self.sample = sample
        self.atom_decoder = atom_decoder
        self.atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
        self.bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
        pass

    def generate_validity_scores(self):
        sampled_s = self.sample
        if sampled_s is None:
            return None
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        molecule_list = []
        for i in range(X.size(0)):
            atom_types = X[i].cpu()
            edge_types = E[i].cpu()
            molecule_list.append([atom_types, edge_types])

        smile_mols = [] 
        for graph in molecule_list:
            atom_types, edge_types = graph
            mol = self.build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
            smile_mols.append(self.check_mol(mol, largest_connected_comp=True))
        binary_list = [0 if mol is None else 1 for mol in smile_mols]
        return torch.tensor(binary_list)

    def build_molecule_with_partial_charges(self, atom_types, edge_types, atom_decoder, verbose=False):
        if verbose:
            print("\nbuilding new molecule")
        bond_dict = self.bond_dict

        mol = Chem.RWMol()
        for atom in atom_types:
            a = Chem.Atom(atom_decoder[atom.item()])
            mol.AddAtom(a)

        edge_types = torch.triu(edge_types)
        all_bonds = torch.nonzero(edge_types)

        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                # not support [O-], [N-], [S-], [NH+] etc.
                flag, atomid_valence = self.check_valency(mol)

                if flag:
                    continue
                else:
                    if len(atomid_valence) == 2:
                        idx = atomid_valence[0]
                        v = atomid_valence[1]
                        an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                        if an in (7, 8, 16) and (v - self.atom_valency[an]) == 1:
                            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                            # print("Formal charge added")
                    else:
                        continue
        return mol
    
    def check_valency(self, mol):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True, None
        except ValueError as e:
            e = str(e)
            p = e.find('#')
            e_sub = e[p:]
            atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
            return False, atomid_valence
        
    def mol2smiles(self, mol):
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)
    
    def check_mol(self, m, largest_connected_comp=True):
        if m is None:
            return None
        try:
            sm = Chem.MolToSmiles(m, isomericSmiles=True)
        except Exception:
            return None
        if largest_connected_comp and '.' in sm:
            vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
            vsm.sort(key=lambda tup: tup[1], reverse=True)
            try:
                mol = Chem.MolFromSmiles(vsm[0][0])
            except:
                return None
        else:
            try:
                mol = Chem.MolFromSmiles(sm)
            except:
                return None
        return mol

