import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ScratchTransformerModel
from config import Config

def generate(model, start_str, char_to_int, int_to_char, max_len=50):
    model.eval()
    input_ids = torch.tensor([char_to_int[c] for c in start_str], dtype=torch.long).unsqueeze(0)
    generated = start_str
    
    with torch.no_grad():
        for _ in range(max_len):
            idx_cond = input_ids[:, -64:] 
            logits = model(idx_cond)
            logits = logits[:, -1, :] / 0.8 # Temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            generated += int_to_char[next_id.item()]
    return generated

def main():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(root_dir, "data", "data.txt")
    model_path = os.path.join(root_dir, "transformer_model.pth")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    cfg = Config(vocab_size=len(chars), dim=256, n_layers=4, n_heads=4, head_dim=64, hidden_dim=512)
    model = ScratchTransformerModel(cfg)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Weights loaded.")
    
    print("\n--- Output ---")
    print(generate(model, "The ", char_to_int, int_to_char))

if __name__ == "__main__":
    main()
