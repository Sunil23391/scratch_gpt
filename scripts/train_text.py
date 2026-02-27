import torch
import os
import sys

# Allow script to find model.py in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ScratchTransformerModel
from config import Config

def main():
    # Path to data relative to this script
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.txt")
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Character-level Tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)

    cfg = Config(
        vocab_size=vocab_size,
        dim=256,
        n_layers=4,
        n_heads=4,
        head_dim=64,
        hidden_dim=512,
        seq_len=64,
        batch_size=16,
        epochs=300
    )

    model = ScratchTransformerModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("Training starting...")
    for epoch in range(cfg.epochs):
        ix = torch.randint(len(data) - cfg.seq_len, (cfg.batch_size,))
        x = torch.stack([data[i:i+cfg.seq_len] for i in ix])
        y = torch.stack([data[i+1:i+cfg.seq_len+1] for i in ix])

        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Save to the root directory
    save_path = os.path.join(os.path.dirname(__file__), "..", "transformer_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
