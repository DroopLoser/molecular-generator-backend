import selfies as sf
import torch

class SelfiesTokenizer:
    def __init__(self, selfies_list):

        chars = set()

        for s in selfies_list:
            tokens = list(sf.split_selfies(s))
            for t in tokens:
                chars.add(t)

        self.token_to_idx = {"<PAD>": 0}

        for ch in sorted(chars):
            self.token_to_idx[ch] = len(self.token_to_idx)

        self.idx_to_token = [
            None
        ] * len(self.token_to_idx)

        for k, v in self.token_to_idx.items():
            self.idx_to_token[v] = k

        self.vocab_size = len(self.token_to_idx)
        self.max_len = max(len(list(sf.split_selfies(s))) for s in selfies_list)

    def encode(self, selfies):
        tokens = list(sf.split_selfies(selfies))

        idxs = [self.token_to_idx[t] for t in tokens]

        while len(idxs) < self.max_len:
            idxs.append(0)

        return torch.tensor(idxs, dtype=torch.long)

    def decode(self, token_ids):
        tokens = []

        for i in token_ids:
            i = int(i)
            if i == 0:
                continue
            tokens.append(self.idx_to_token[i])

        return sf.join_selfies(tokens)