import random
import mmap
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper-parameters
# Device setting (GPU if available, CPU if not)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
batch_size = 128  # How many sequences we want running at the same time (32)
block_size = 32  # Sequence length (128)
max_iters = 300  # Number of training iterations
# Learning rate (updated by the AdamW optimization algorithm)
learning_rate = 3e-4
eval_iters = 100  # Number of evaluation iterations, reporting the loss
n_embd = 384  # Number of total dimensions
n_head = 8  # Number of heads running (in parallel)
n_layer = 8  # Number of decoder blocks
dropout = 0.2  # Dropout percentage of neurons

chars = ""
with open('text_files/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(s): return [string_to_int[c] for c in s]
def decode(l): return ''.join([int_to_string[i] for i in l])


class Head(nn.Module):
    # One head of self-attention, scaled dot-product attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Registers the no-look ahead masking in the model state
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Input of size (batch, time-step, channels)
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # Compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float(
            '-inf'))  # (B, T, T), prevents look-head
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out  # Output of size (batch, time-step, head size)


class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Allows for 4 heads to run in parallel within the ModuleList
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projects the head_size * num_heads to n_embd, adding another learnable parameter with a weight and bias
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate each head together along their last dimension (B, T, C(F)) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Drops out 20% of the network's neurons to prevent overfitting
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # A simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # Non-linearity activation, changes all inputs below 0 to an output of 0 and leaves all positive values the same
            # Ensures that output shape is (n_embd, n_embd) by aligning sizes for multiplication
            nn.Linear(4 * n_embd, n_embd),
            # Drops out a 20% of the network's neurons to prevent overfitting
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # Transformer block -> communication followed by computation
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)  # Self attention
        x = self.ln1(x + y)  # Add and normalize
        y = self.ffwd(x)  # Feed forward
        x = self.ln2(x + y)  # Add and normalize
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd)  # Embeddings embedding (lookup) table
        # Positional encoding embedding (lookup) table
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # Creates 4 decoder layers that run synchronously, with each block depending on the completion and input of the previous block
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    # Initializes weights around specific standard deviations
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(index)  # (B, T, C)
        pos_emb = self.positional_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # Index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens
            index_cond = index[:, -block_size:]
            # Get the predictions
            logits, loss = self.forward(index_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


model = GPTLanguageModel(vocab_size)

print('Loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Loaded successfully.')

m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(
        context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
