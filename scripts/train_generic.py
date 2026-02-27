import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from model import ScratchTransformerModel
from config import Config

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, cfg):
    setup(rank, world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Initialize model and wrap with DDP
    model = ScratchTransformerModel(cfg).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler() # For mixed precision
    
    # Placeholder for actual distributed DataLoader (e.g., from S3 or large local dataset)
    # loader = ... 
    
    model.train()
    # Training Loop
    for epoch in range(cfg.epochs):
        # for batch in loader:
        #     optimizer.zero_grad()
        #     with autocast(device_type='cuda', dtype=torch.float16):
        #         inputs = batch[:, :-1].to(device)
        #         targets = batch[:, 1:].to(device)
        #         logits = model(inputs)
        #         loss = torch.nn.functional.cross_entropy(
        #             logits.reshape(-1, cfg.vocab_size), 
        #             targets.reshape(-1)
        #         )
        #     
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        
        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch+1} completed placeholder")

    cleanup()

if __name__ == "__main__":
    cfg = Config()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
    else:
        print("Distributed training requires at least 2 GPUs. Skipping DDP spawn.")
