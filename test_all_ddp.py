#!/usr/bin/env python3
"""
Test runner for all DDP implementations
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_ddp_test(ddp_type, world_size=2, steps=10, batch_size=32, backend="gloo"):
    """Run a single DDP test"""
    print(f"\n{'='*60}")
    print(f"Testing {ddp_type.upper()} DDP Implementation")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "test_minimal_ddp.py",
        "--ddp-type", ddp_type,
        "--world-size", str(world_size),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--backend", backend
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {ddp_type.upper()} DDP test PASSED")
            print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
            return True, end_time - start_time
        else:
            print(f"❌ {ddp_type.upper()} DDP test FAILED")
            print(f"Error output: {result.stderr}")
            return False, end_time - start_time
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {ddp_type.upper()} DDP test TIMEOUT")
        return False, 60.0
    except Exception as e:
        print(f"💥 {ddp_type.upper()} DDP test ERROR: {e}")
        return False, 0.0

def test_all_implementations(world_size=2, steps=10, batch_size=32, backend="gloo"):
    """Test all DDP implementations"""
    implementations = ["naive", "overlap", "overlap_bucket"]
    results = {}
    
    print(f"Testing all DDP implementations")
    print(f"World size: {world_size}, Steps: {steps}, Batch size: {batch_size}, Backend: {backend}")
    
    for ddp_type in implementations:
        success, execution_time = run_ddp_test(ddp_type, world_size, steps, batch_size, backend)
        results[ddp_type] = {
            "success": success,
            "execution_time": execution_time
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for ddp_type, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        time_str = f"{result['execution_time']:.2f}s" if result["success"] else "N/A"
        print(f"{ddp_type.upper():15} | {status:8} | {time_str:>8}")
        if not result["success"]:
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("🎉 All DDP implementations passed!")
        return True
    else:
        print("⚠️  Some DDP implementations failed!")
        return False

def benchmark_implementations(world_size=2, steps=50, batch_size=64, backend="gloo", runs=3):
    """Benchmark all implementations with multiple runs"""
    implementations = ["naive", "overlap", "overlap_bucket"]
    benchmark_results = {}
    
    print(f"Benchmarking DDP implementations")
    print(f"World size: {world_size}, Steps: {steps}, Batch size: {batch_size}, Backend: {backend}")
    print(f"Running each implementation {runs} times")
    
    for ddp_type in implementations:
        print(f"\nBenchmarking {ddp_type.upper()} DDP...")
        times = []
        successes = 0
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...", end=" ")
            success, execution_time = run_ddp_test(ddp_type, world_size, steps, batch_size, backend)
            if success:
                times.append(execution_time)
                successes += 1
                print(f"✅ {execution_time:.2f}s")
            else:
                print("❌")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            benchmark_results[ddp_type] = {
                "success_rate": successes / runs,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            }
        else:
            benchmark_results[ddp_type] = {
                "success_rate": 0,
                "avg_time": float('inf'),
                "min_time": float('inf'),
                "max_time": float('inf'),
                "times": []
            }
    
    # Print benchmark results
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Implementation':15} | {'Success Rate':12} | {'Avg Time':10} | {'Min Time':10} | {'Max Time':10}")
    print(f"{'-'*80}")
    
    for ddp_type, result in benchmark_results.items():
        success_rate = f"{result['success_rate']*100:.0f}%"
        avg_time = f"{result['avg_time']:.2f}s" if result['avg_time'] != float('inf') else "N/A"
        min_time = f"{result['min_time']:.2f}s" if result['min_time'] != float('inf') else "N/A"
        max_time = f"{result['max_time']:.2f}s" if result['max_time'] != float('inf') else "N/A"
        
        print(f"{ddp_type.upper():15} | {success_rate:12} | {avg_time:10} | {min_time:10} | {max_time:10}")
    
    print(f"{'='*80}")
    
    return benchmark_results

def main():
    parser = argparse.ArgumentParser(description="Test and benchmark DDP implementations")
    parser.add_argument("--mode", choices=["test", "benchmark"], default="test",
                       help="Test mode: 'test' for basic testing, 'benchmark' for performance comparison")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Total batch size")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for benchmarking")
    parser.add_argument("--ddp-type", type=str, choices=["naive", "overlap", "overlap_bucket"],
                       help="Test only specific DDP implementation")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    
    if args.ddp_type:
        # Test only specific implementation
        print(f"Testing only {args.ddp_type.upper()} DDP implementation")
        success, execution_time = run_ddp_test(args.ddp_type, args.world_size, args.steps, args.batch_size, args.backend)
        sys.exit(0 if success else 1)
    
    if args.mode == "test":
        success = test_all_implementations(args.world_size, args.steps, args.batch_size, args.backend)
        sys.exit(0 if success else 1)
    else:  # benchmark mode
        benchmark_results = benchmark_implementations(args.world_size, args.steps, args.batch_size, args.backend, args.runs)
        
        # Find fastest implementation
        fastest = None
        fastest_time = float('inf')
        for ddp_type, result in benchmark_results.items():
            if result['success_rate'] > 0 and result['avg_time'] < fastest_time:
                fastest = ddp_type
                fastest_time = result['avg_time']
        
        if fastest:
            print(f"\n🏆 Fastest implementation: {fastest.upper()} ({fastest_time:.2f}s average)")
        else:
            print(f"\n⚠️  No successful implementations found")

if __name__ == "__main__":
    main()









