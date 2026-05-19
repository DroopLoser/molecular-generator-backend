##%%writefile generate.py
"""
Clean molecule generation script (no filtering).

Features:
1. VAE + GNN latent optimization
2. Accuracy per molecule
3. Saves molecule images
4. Shows top-k results
"""

import argparse
import os
import torch

from rdkit import Chem
from rdkit.Chem.Draw import MolToFile

from models.vae import SmilesVAE
from models.gnn import MolecularGNN
from utils.optimization import optimize_latent_space


# ======================================================
# Load VAE
# ======================================================
def load_vae(device):
    checkpoint = torch.load("vae.pt", map_location=device)

    model = SmilesVAE(
        vocab_size=checkpoint["vocab_size"],
        max_len=checkpoint["max_len"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_token = checkpoint["idx_to_token"]
    return model, idx_to_token


# ======================================================
# Load GNN
# ======================================================
def load_gnn(device):
    checkpoint = torch.load("gnn.pt", map_location=device)

    property_names = checkpoint.get("property_names", ["MolWt"])

    model = MolecularGNN(
        in_channels=checkpoint.get("in_channels", 1),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        out_dim=len(property_names),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, property_names


# ======================================================
# Accuracy
# ======================================================
def compute_accuracy(target, predicted):
    error = abs(predicted - target)

    if abs(target) < 1e-6:
        return max(0.0, 100.0 - error * 100.0)

    return max(0.0, 100.0 * (1 - error / abs(target)))


# ======================================================
# Save image
# ======================================================
def save_molecule_image(smiles, filename):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    MolToFile(mol, filename, size=(400, 400))
    return True


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--property", dest="selected_property", type=str, default="QED")
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    print("Loading VAE...")
    vae_model, idx_to_token = load_vae(device)

    print("Loading GNN...")
    gnn_model, property_names = load_gnn(device)

    if args.selected_property not in property_names:
        raise ValueError(f"Unsupported property: {args.selected_property}")

    print("\nGenerating molecules...")
    print("Target:", args.target)

    # ======================================================
    # GENERATION
    # ======================================================
    results = optimize_latent_space(
        vae_model=vae_model,
        gnn_model=gnn_model,
        idx_to_token=idx_to_token,
        target_property=args.target,
        selected_property=args.selected_property,
        num_samples=args.samples,
        device=device,
    )

    if not results:
        print("No molecules generated.")
        return

    # ======================================================
    # SCORE + SORT
    # ======================================================
    for r in results:
        r["accuracy"] = compute_accuracy(
            args.target,
            r["predicted_property"]
        )

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    top_results = results[: args.top_k]

    # ======================================================
    # OUTPUT
    # ======================================================
    os.makedirs("generated_images", exist_ok=True)

    print("\nTop Molecules")
    print("=" * 80)

    for i, r in enumerate(top_results, 1):
        smiles = r["smiles"]
        path = f"generated_images/mol_{i}.png"

        save_molecule_image(smiles, path)

        print(
            f"{i}. SMILES: {smiles} | "
            f"{r['selected_property']}: {r['predicted_property']:.4f} | "
            f"Error: {r['error']:.4f} | "
            f"Accuracy: {r['accuracy']:.2f}%"
        )
        print("   Image:", path)

    print("\nDone.")


if __name__ == "__main__":
    main()