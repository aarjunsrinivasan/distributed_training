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

## Command Line Options

### test_minimal_ddp.py Options

- `--ddp-type`: Choose DDP implementation (`naive`, `overlap`, `overlap_bucket`)
- `--world-size`: Number of processes (default: 2)
- `--steps`: Number of training steps (default: 20)
- `--batch-size`: Total batch size (default: 32)
- `--backend`: Distributed backend (`nccl` for GPU, `gloo` for CPU)

### test_all_ddp.py Options

- `--mode`: Test mode (`test` for basic testing, `benchmark` for performance comparison)
- `--world-size`: Number of processes (default: 2)
- `--steps`: Number of training steps (default: 10 for test, 50 for benchmark)
- `--batch-size`: Total batch size (default: 32 for test, 64 for benchmark)
- `--backend`: Distributed backend (default: `gloo`)
- `--runs`: Number of runs for benchmarking (default: 3)
- `--ddp-type`: Test only specific implementation

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

## Expected Output

### Single Implementation Test

```
Testing NAIVE DDP implementation
World size: 2, Steps: 20, Batch size: 32, Backend: gloo
Running NAIVE DDP with rank 0 and world size 2 and backend gloo
Rank 0: Using NaiveDDP implementation
Running NAIVE DDP with rank 1 and world size 2 and backend gloo
Rank 1: Using NaiveDDP implementation
...
Rank 0: Verification successful! NAIVE DDP parameters match baseline.
```

### All Implementations Test

```
Testing all DDP implementations
World size: 2, Steps: 10, Batch size: 32, Backend: gloo

============================================================
Testing NAIVE DDP Implementation
============================================================
✅ NAIVE DDP test PASSED
⏱️  Execution time: 2.34 seconds

============================================================
Testing OVERLAP DDP Implementation
============================================================
✅ OVERLAP DDP test PASSED
⏱️  Execution time: 1.87 seconds

============================================================
Testing OVERLAP_BUCKET DDP Implementation
============================================================
✅ OVERLAP_BUCKET DDP test PASSED
⏱️  Execution time: 1.92 seconds

============================================================
TEST SUMMARY
============================================================
NAIVE          | ✅ PASS   |    2.34s
OVERLAP        | ✅ PASS   |    1.87s
OVERLAP_BUCKET | ✅ PASS   |    1.92s
============================================================
🎉 All DDP implementations passed!
```

## Troubleshooting

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

### Debug Mode

For more verbose output, you can modify the test files to add more print statements or use Python's debugger:

```bash
python -m pdb test_minimal_ddp.py --ddp-type naive
```

## Implementation Details

### NaiveDDP
- Performs all-reduce on each parameter gradient individually
- Simple but potentially slower due to many small communications

### DDPOverlap
- Overlaps gradient computation with communication
- Uses asynchronous gradient synchronization

### DDPOverlapBucket
- Groups parameters into buckets for more efficient communication
- Reduces the number of all-reduce operations

## Performance Considerations

- **CPU vs GPU**: Use `gloo` backend for CPU, `nccl` for GPU
- **World size**: Larger world sizes may show more performance differences
- **Batch size**: Larger batches may better demonstrate communication overhead
- **Steps**: More steps provide more stable performance measurements

## Adding New Implementations

To add a new DDP implementation:

1. Add the implementation to `minimal_ddp.py`
2. Add the new type to the choices in `test_minimal_ddp.py`
3. Add the implementation logic in the `run()` function
4. Update `test_all_ddp.py` to include the new implementation


