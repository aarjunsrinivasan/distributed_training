import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    def __init__(self, module):
        """
        A simplified DDP implementation.
        
        Args:
            module: The model to parallelize
            world_size: Number of parallel processes
        """
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # --- Step 1: Broadcast parameters at startup ---
        # Ensure all ranks start with the same parameters
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)  # Rank 0 sends to all others

        # --- Step 2: Register gradient hooks ---
        # These hooks will automatically synchronize gradients during backward pass
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(self._make_allreduce_hook(p))

    def _make_allreduce_hook(self, p):
        """Create a hook that performs all-reduce on gradients."""
        def hook(grad):
            # Sum gradients across all processes
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            # Average the gradients (simulating a single large batch)
            return grad / self.world_size
        return hook
    
    def forward(self, *args, **kwargs):
        """Simply forward through the wrapped module."""
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        pass





class DDPOverlap(nn.Module):
    """
    A minimal Distributed Data‑Parallel wrapper that overlaps gradient communication with the
    computation of the backward pass by immediately launching an asynch `all_reduce` on each
    parameter’s gradient as soon as it is produced.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.handles: list[tuple[dist.Work, torch.nn.Parameter]] = []
        self.world_size = dist.get_world_size()

        # Broadcast parameters from rank 0 to all other ranks
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        # Register post-accumulate gradient hook for each parameter
        def make_hook(param: torch.nn.Parameter):
            def hook(*_: torch.Tensor):
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, param))

            return hook

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(make_hook(p))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed."""
        for work, param in self.handles:
            work.wait()
            param.grad.div_(self.world_size)
        self.handles.clear()



class DDPOverlapBucket(nn.Module):
    """
    bucketed, overlap DDP wrapper that:
      • broadcasts params once
      • groups params into buckets
      • launches async all-reduce when a bucket’s grads are ready
      • waits/unflattens/averages at sync()
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # --- Broadcast initial parameters from rank 0 ---
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # --- Build buckets (fixed order, sized by num elements) ---
        dtype = next(self.module.parameters()).dtype
        bytes_per_param = dtype.itemsize
        bucket_cap = int(bucket_size_mb * 1024**2 / bytes_per_param)

        self.buckets = []  # each: {"params": [...], "need": int, "ready": int, "handle": None, ...}

        cur = {"params": [], "need": 0, "ready": 0, "handle": None,
               "flat": None, "grads": None} # bucket state

        # reverse order ≈ backward visitation order
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            n = p.numel()
            if cur["need"] > 0 and cur["need"] + n > bucket_cap:
                self.buckets.append(cur)
                cur = {"params": [], "need": 0, "ready": 0, "handle": None,
                       "flat": None, "grads": None}  # bucket state zero out after it is filled
            cur["params"].append(p)
            cur["need"] += n
        if cur["need"] > 0: # add the last bucket
            self.buckets.append(cur)

        # mark each param with its bucket index and add a post-accumulate hook
        for b_idx, bucket in enumerate(self.buckets):
            for p in bucket["params"]:
                p._bucket_idx = b_idx  # simple tag
                p.register_post_accumulate_grad_hook(self._make_hook(p))

        # track pending buckets to finalize at sync()
        self._pending = []

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(*_):
            b = self.buckets[param._bucket_idx]
            b["ready"] += param.numel() # number of grads in this bucket that are ready to be all-reduced

            # If the whole bucket's grads are now ready → flatten + async all-reduce
            if b["ready"] == b["need"]:
                grads = [p.grad for p in b["params"] if p.grad is not None]
                if len(grads) == 0:
                    # reset for next iteration and skip
                    b["ready"] = 0
                    return
                flat = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
                # remember what to unflatten into and the handle to wait on
                b["grads"] = grads
                b["flat"] = flat
                b["handle"] = handle
                self._pending.append(b)
                # reset ready counter for next iteration
                b["ready"] = 0
        return hook

    def forward(self, *a, **kw):
        return self.module(*a, **kw)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        """
        Wait for in-flight all-reduces, average, and scatter back to .grad tensors.
        Call once per iteration AFTER loss.backward() and BEFORE optimizer.step().
        """
        for b in self._pending:
            b["handle"].wait()
            b["flat"].div_(self.world_size)
            unflat = torch._utils._unflatten_dense_tensors(b["flat"], b["grads"])
            for g_dst, g_src in zip(b["grads"], unflat):
                g_dst.copy_(g_src)
            # clear transient state
            b["handle"] = None
            b["flat"] = None
            b["grads"] = None
        self._pending.clear()

