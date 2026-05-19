#%%writefile models/vae.py
"""
Character-level SMILES Variational Autoencoder (VAE).

This model:
1. Encodes a tokenized SMILES string into a latent vector z.
2. Samples from the latent distribution.
3. Decodes z back into a sequence of token logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmilesVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim=128,
        hidden_dim=256,
        latent_dim=64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        # Encoder
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        self.fc_logvar = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        # Decoder input projection
        self.fc_decode = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Output projection
        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------
    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder(emb)
        h = h[-1]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    # --------------------------------------------------
    # Reparameterization Trick
    # --------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------
    def decode(self, z):
        # Initial hidden state
        h0 = self.fc_decode(z).unsqueeze(0)

        # Repeat the latent-derived hidden state as input
        decoder_input = (
            h0.squeeze(0)
            .unsqueeze(1)
            .repeat(1, self.max_len, 1)
        )

        out, _ = self.decoder(
            decoder_input,
            h0,
        )

        logits = self.output_layer(out)

        return logits

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


# ------------------------------------------------------
# Loss Function
# ------------------------------------------------------
def vae_loss(
    logits,
    targets,
    mu,
    logvar,
    beta=0.01,
):
    """
    Reconstruction loss + KL divergence.
    """
    # Reconstruction loss
    recon_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=0,
    )

    # KL divergence
    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl, recon_loss, kl