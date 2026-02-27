import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from model import ScratchTransformerModel
from config import Config

def main():
    # Load configuration
    cfg = Config(
        dim=128, 
        n_layers=2, 
        n_heads=4, 
        hidden_dim=256,
        batch_size=4,
        epochs=5
    )
    
    # Initialize model
    model = ScratchTransformerModel(cfg)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # Mock data generation
    # simulating a .pt file with tokenized data
    mock_data_path = "mock_data.pt"
    if not os.path.exists(mock_data_path):
        data = torch.randint(0, cfg.vocab_size, (16, cfg.seq_len))
        torch.save(data, mock_data_path)
    
    data = torch.load(mock_data_path)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0
        for i, (batch,) in enumerate(loader):
            # Input: tokens [0 to N-1], Target: tokens [1 to N]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            
            # Flatten for cross-entropy
            loss = criterion(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg_loss:.4f}")

    # Cleanup mock data
    if os.path.exists(mock_data_path):
        os.remove(mock_data_path)

if __name__ == "__main__":
    main()
