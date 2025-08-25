import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # number of independent sequences to be processed in parallel
block_size = 256 # number of characters used in context for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb_d = 384 # number of embeddings dimensions
num_heads = 6
head_size = n_emb_d
num_layers = 6
dropout = 0.2 # 20% of neurons dropped

torch.manual_seed(42)

# open, read, and close text file
with open('/users/salah/documents/mini-gpt/data/tiny-shakespeare.txt', 'r', 
          encoding='utf-8') as file:
    text = file.read()

# get unique characters from text
chrs = sorted(list(set(text)))
vocab_size = len(chrs)

# map characters to integers
chr_to_int = {ch: i for i, ch in enumerate(chrs)}
  
# map integers to characters  
int_to_chr = {i: ch for i, ch in enumerate(chrs)}
    
def encode(text: str) -> list[int]:
    """Tokenize characters in text to integers.

    Args:
        text (str): Any string.
    Returns:
        list[int]: Token list of integers.
    """
    return [chr_to_int[ch] for ch in text]
    

def decode(tokens: list[int]) -> str:
    """Revert token integers to text.

    Args:
        tokens (list[int]): List of token integers.

    Returns:
        str: Text derived from tokens.
    """
    return ''.join([int_to_chr[i] for i in tokens])

data = torch.tensor(encode(text), dtype=torch.long)
# split token data into train & validation sets
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

def get_batch(data: torch.tensor, batch_size: int, block_size: int) -> tuple[torch.tensor]:
    """Returns 

    Args:
        data (torch.tensor): Token tensor.
        batch_size (int): Number of batches of contexts to be processed.
        block_size (int): Length of the context of characters.

    Returns:
        tuple[torch.tensor]: Stacked context and target tensors of dimension 
        block_size x block_size. 
    """
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,))
    j, k = [], []
    for i in ix:
        j.append(data[i:i+block_size])
        k.append(data[i+1:i+block_size+1])
    x = torch.stack(j)
    y = torch.stack(k)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Return the average loss for train and validation datasets over eval_iters iterations.
    """
    out = {}
    model.eval = ()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train if (split == 'train') else val, batch_size=batch_size, 
                             block_size=block_size)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
class Head(nn.Module):
    """Single head of self attention.

    Superclass: nn.Module
    """
    def __init__(self, n_emb_d: int, head_size: int, dropout: float) -> None:
        """Initialize a single head.

        Args:
            n_emb_d (int): _description_
            head_size (int): _description_
            dropout (float): _description_
        """
        super().__init__()
        self.key = nn.Linear(n_emb_d, head_size, bias=False)
        self.query = nn.Linear(n_emb_d, head_size, bias=False)
        self.value = nn.Linear(n_emb_d, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            torch.tensor: _description_
        """
        b, t, c = x.shape
        k = self.key(x) # (b x t x c)
        q = self.query(x) # (b x t x c)
        v = self.value(x) # (b x t x c)
        
        # compute normalized attention scores
        weights = q @ k.transpose(-2, -1) * c**-0.5 # (b x t x c) @ (b x c x t) -> (b x t x t)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float('-inf')) # (b x t x t)
        weights = F.softmax(weights, dim=-1) # (b x t x t)
        weights = self.dropout(weights)
        out = weights @ v # (b x t x t) @ (b x t x c) -> (b x t x c)
        return out

class MultiHeadAttention(nn.Module):
    """Multiples heads of self-attention in parallel.

    Superclass: nn.Module
    """
    def __init__(self, num_heads: int, head_size: int, n_emb_d: int, dropout: float) -> None:
        """Initialize multihead self-attention.

        Args:
            num_head (int): _description_
            head_size (int): _description_
            n_emb_d (int): _description_
            dropout (float): _description_

        Returns:
            _type_: _description_
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb_d=n_emb_d, head_size=head_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb_d, n_emb_d)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            torch.tensor: _description_
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity.

    Superclass: nn.Module
    """
    def __init__(self, n_emb_d: int, dropout:int) -> None:
        """_summary_

        Args:
            n_emb_d (int): _description_
            dropout (int): _description_

        Returns:
            _type_: _description_
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb_d, 4 * n_emb_d), 
            nn.ReLU(),
            nn.Linear(4 * n_emb_d, n_emb_d), 
            nn.Dropout(dropout)
            )
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            torch.tensor: _description_
        """
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation.

    Superclass: nn.Module
    """
    def __init__(self, n_emb_d: int, num_heads: int) -> None:
        """Initialize transformer block.

        Args:
            n_emb_d (int): Embeddings dimension.
            num_head (int): Number of heads.
        """
        super().__init__()
        head_size = n_emb_d // num_heads
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, n_emb_d=n_emb_d, dropout=dropout)
        self.ffwd = FeedForward(n_emb_d=n_emb_d, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_emb_d)
        self.ln2 = nn.LayerNorm(n_emb_d)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            torch.tensor: _description_
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x

class BigramLanguageModel(nn.Module):
    """_summary_

    Superclass: nn.Module
    """
    
    def __init__(self, vocab_size: int, n_emb_d: int, num_layers: int) -> None:
        """Initialize a BigramLanguageModel.

        Args:
            vocab_size (int): _description_
            n_emb_d (int): _description_
            num_layers (int): _description_
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb_d)
        self.position_embedding_table = nn.Embedding(block_size, n_emb_d)
        self.blocks = nn.Sequential(*[Block(n_emb_d=n_emb_d, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_emb_d)
        self.lm_head = nn.Linear(n_emb_d, vocab_size) # language modelling head
        
    def forward(self, idx: torch.tensor, targets: torch.tensor=None) -> tuple:
        """_summary_

        Args:
            idx (_type_): _description_
            targets (_type_): _description_
            
        Returns:
            tuple: _description_
        """ 
        b, t = idx.shape
        
        # idx & targets are batch x time dimensional tensor
        tok_emb = self.token_embedding_table(idx) # <- (batch x time x channel) dimensional
        pos_emb = self.position_embedding_table(torch.arange(t, device=device)) # (T, C)
        x = tok_emb + pos_emb # b x t x c
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch x time x vocab size)
        
        if targets == None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets) # <- expects (batch x time) dimensional only
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """_summary_

        Args:
            idx (_type_): _description_
            max_new_tokens (_type_): _description_
        """
        # idx is b x t array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (b x c)
            # apply Softmax to get probablities
            probs = F.softmax(logits, dim=1) # (b x c)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (b x 1)
            # append sampled idx to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (b x t+1)
        return idx

model = BigramLanguageModel(vocab_size, n_emb_d, num_layers)
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # Once in a while evaluate train and val losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    
    # sample a batch of data
    x_train, y_train = get_batch(data=train, batch_size=batch_size, block_size=block_size)
    
    # evaluate the loss
    logits, loss = m(x_train, y_train)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.ones((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))