import os
import torch
import time
import argparse

def allocate_gpu_memory(size_mb):
    """Allocate GPU memory of a specified size in MB."""
    tensor_size = size_mb * 1024 * 1024 // 4  # Convert MB to number of float32 elements
    try:
        _ = torch.zeros(tensor_size, dtype=torch.float32, device='cuda')
        print(f"Allocated {size_mb} MB on the GPU.")
    except RuntimeError as e:
        print(f"Failed to allocate memory: {e}")

def main(run_time_minutes, gpu_id, memory_size_mb):
    """Main function to allocate GPU memory and run for a specified time."""
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

    # Allocate GPU memory
    allocate_gpu_memory(memory_size_mb)

    # Run for the specified time, printing every 5 minutes
    run_time_seconds = run_time_minutes * 60
    interval = 5 * 60  # 5 minutes in seconds
    start_time = time.time()
    end_time = start_time + run_time_seconds

    while time.time() < end_time:
        remaining_time = int((end_time - time.time()) / 60)
        print(f"Time left: {remaining_time} minutes")
        time.sleep(min(interval, end_time - time.time()))

    print("Terminating script.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Memory Allocation Script")
    parser.add_argument("--runtime_minutes", type=int, required=True, help="Runtime in minutes")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--memory_size_mb", type=int, required=True, help="Memory size to allocate in MB")

    args = parser.parse_args()

    main(args.runtime_minutes, args.gpu_id, args.memory_size_mb)
