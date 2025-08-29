# DDP Implementation Testing Guide

This guide explains how to use the modified `test_minimal_ddp.py` to test different DDP implementations.

## Available DDP Implementations

The test system supports three DDP implementations:

1. **NaiveDDP** (`naive`) - Basic DDP implementation with per-parameter gradient synchronization
2. **DDPOverlap** (`overlap`) - DDP with gradient computation and communication overlap
3. **DDPOverlapBucket** (`overlap_bucket`) - DDP with bucketed gradient synchronization

## Quick Start

### Test a Single Implementation

```bash
# Test naive DDP
python test_minimal_ddp.py --ddp-type naive

# Test overlap DDP
python test_minimal_ddp.py --ddp-type overlap

# Test overlap bucket DDP
python test_minimal_ddp.py --ddp-type overlap_bucket
```

### Test All Implementations

```bash
# Basic testing of all implementations
python test_all_ddp.py --mode test

# Benchmark all implementations
python test_all_ddp.py --mode benchmark
```



## Examples

### Basic Testing

```bash
# Test naive DDP with 4 processes
python test_minimal_ddp.py --ddp-type naive --world-size 4

# Test overlap DDP with custom parameters
python test_minimal_ddp.py --ddp-type overlap --steps 50 --batch-size 64

# Test on GPU (if available)
python test_minimal_ddp.py --ddp-type overlap_bucket --backend nccl
```

### Performance Benchmarking

```bash
# Quick benchmark
python test_all_ddp.py --mode benchmark --steps 20 --runs 2

# Comprehensive benchmark
python test_all_ddp.py --mode benchmark --world-size 4 --steps 100 --batch-size 128 --runs 5
```

### Test Specific Implementation

```bash
# Test only naive DDP
python test_all_ddp.py --ddp-type naive

# Test only overlap DDP with custom parameters
python test_all_ddp.py --ddp-type overlap --world-size 3 --steps 30
```


### Common Issues

1. **Port already in use**: Change the `MASTER_PORT` environment variable
   ```bash
   export MASTER_PORT=12358
   python test_minimal_ddp.py --ddp-type naive
   ```

2. **CUDA not available**: Tests will automatically fall back to CPU
   ```bash
   python test_minimal_ddp.py --ddp-type naive --backend gloo
   ```

3. **Batch size not divisible by world size**: Ensure batch size is divisible by world size
   ```bash
   # This will work
   python test_minimal_ddp.py --batch-size 32 --world-size 2
   
   # This will fail
   python test_minimal_ddp.py --batch-size 31 --world-size 2
   ```


## Adding New Implementations

To add a new DDP implementation:

1. Add the implementation to `minimal_ddp.py`
2. Add the new type to the choices in `test_minimal_ddp.py`
3. Add the implementation logic in the `run()` function
4. Update `test_all_ddp.py` to include the new implementation


