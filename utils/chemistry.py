#%%writefile utils/chemistry.py
"""
Chemistry helper functions for validation, descriptor calculation,
and molecule rendering.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.Draw import MolToFile


# ------------------------------------------------------
# Validation
# ------------------------------------------------------
def is_valid_smiles(smiles):
    """
    Return True if the SMILES string can be parsed by RDKit.
    """
    if not smiles:
        return False

    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

import base64

def mol_to_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Chem.Draw.MolToImage(mol)

    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def canonicalize_smiles(smiles):
    """
    Convert a SMILES string to canonical form.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


# ------------------------------------------------------
# Descriptor Calculation
# ------------------------------------------------------
def property_by_name(smiles, property_name="MolWt"):
    """
    Compute a selected molecular property.
    Supported properties:
        MolWt, LogP, TPSA, HBD, HBA, QED
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    property_name = property_name.lower()

    if property_name == "molwt":
        return Descriptors.MolWt(mol)

    elif property_name == "logp":
        return Descriptors.MolLogP(mol)

    elif property_name == "tpsa":
        return Descriptors.TPSA(mol)

    elif property_name == "hbd":
        return Descriptors.NumHDonors(mol)

    elif property_name == "hba":
        return Descriptors.NumHAcceptors(mol)

    elif property_name == "qed":
        return QED.qed(mol)

    else:
        raise ValueError(
            "Unsupported property. "
            "Use MolWt, LogP, TPSA, HBD, HBA, or QED."
        )


# ------------------------------------------------------
# Molecule Drawing
# ------------------------------------------------------
def draw_molecule(smiles, filename="molecule.png"):
    """
    Save a 2D image of the molecule.
    Returns True if successful.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False

    MolToFile(mol, filename)
    return True