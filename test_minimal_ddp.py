import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from minimal_ddp import DDPOverlap, NaiveDDP, DDPOverlapBucket


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> str:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")

    if torch.cuda.is_available() and backend == "nccl":
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_ddp() -> None:
    dist.barrier()
    dist.destroy_process_group()



class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 10, bias=False)
        self.fc4.weight = self.fc2.weight
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x



def run(rank: int, world_size: int, steps: int = 20, batch_size: int = 32, ddp_type: str = "naive", backend: str = "nccl"):
    print(f"Running {ddp_type.upper()} DDP with rank {rank} and world size {world_size} and backend {backend}")

    device = setup_ddp(rank, world_size, backend)
    dist.barrier()

    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    local_batch_size = batch_size // world_size

    # Create base model
    base_model = ToyModelWithTiedWeights().to(device)
    
    # Wrap with selected DDP implementation
    if ddp_type.lower() == "naive":
        model = NaiveDDP(base_model)
        print(f"Rank {rank}: Using NaiveDDP implementation")
    elif ddp_type.lower() == "overlap":
        model = DDPOverlap(base_model)
        print(f"Rank {rank}: Using DDPOverlap implementation")
    elif ddp_type.lower() == "overlap_bucket":
        model = DDPOverlapBucket(base_model)
        print(f"Rank {rank}: Using DDPOverlapBucket implementation")
    else:
        raise ValueError(f"Unknown DDP type: {ddp_type}. Available options: naive, overlap, overlap_bucket")
    
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()


    # Baseline setup for rank 0
    baseline_model = None
    baseline_optim = None
    if rank == 0:
        baseline_model = ToyModelWithTiedWeights().to(device)
        # Ensure baseline starts with identical parameters as the DDP model after broadcast
        baseline_model.load_state_dict(model.module.state_dict())
        baseline_optim = torch.optim.SGD(baseline_model.parameters(), lr=1e-2)

    for step in range(steps):
        # Deterministic Data Generation
        data_seed = step
        torch.manual_seed(data_seed)
        all_x = torch.randn(batch_size, 32, device=device)
        all_y = torch.randint(0, 10, (batch_size,), device=device)

        # DDP Training Step with partition for the current rank
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size
        x = all_x[start_idx:end_idx]
        y = all_y[start_idx:end_idx]

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Handle different DDP implementations
        if hasattr(model, 'finish_gradient_synchronization'):
            model.finish_gradient_synchronization()
        # For naive DDP, gradients are already synchronized during backward


        optim.step()

        # Baseline Training Step on rank 0 only
        if rank == 0:
            baseline_optim.zero_grad(set_to_none=True)
            baseline_logits = baseline_model(all_x)  # Use full batch
            baseline_loss = loss_fn(baseline_logits, all_y)
            baseline_loss.backward()
            baseline_optim.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)

    # Verification vs single-process baseline on rank 0
    if rank == 0:
        print("Rank 0: Verifying DDP model parameters against single-process baseline...")
        
        # Get the actual model parameters (handle different wrapper types)
        ddp_model_params = model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters()
        
        for (name, param), (base_name, base_param) in zip(ddp_model_params, baseline_model.named_parameters()):
            assert name == base_name
            assert torch.allclose(param.data, base_param.data), (
                f"Mismatch found in parameter: {name}. Max diff: {torch.abs(param.data - base_param.data).max().item()}"
            )
        
        print(f"Rank 0: Verification successful! {ddp_type.upper()} DDP parameters match baseline.")

    cleanup_ddp()




def _parse_args():
    parser = argparse.ArgumentParser(description="Test different DDP implementations")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Total batch size")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend (nccl/gloo)")
    parser.add_argument("--ddp-type", type=str, default="naive", 
                       choices=["naive", "overlap", "overlap_bucket"],
                       help="DDP implementation to test")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    WORLD_SIZE = args.world_size

    assert args.batch_size % WORLD_SIZE == 0, (
        f"Batch size ({args.batch_size}) must be divisible by world size ({WORLD_SIZE})"
    )

    BACKEND = args.backend or "nccl" if torch.cuda.is_available() else "gloo"
    
    print(f"Testing {args.ddp_type.upper()} DDP implementation")
    print(f"World size: {WORLD_SIZE}, Steps: {args.steps}, Batch size: {args.batch_size}, Backend: {BACKEND}")
    
    mp.spawn(run, args=(WORLD_SIZE, args.steps, args.batch_size, args.ddp_type, BACKEND), nprocs=WORLD_SIZE, join=True)
