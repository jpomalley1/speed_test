import tensorflow as tf
import time
import numpy as np

def benchmark_tf_matmul(N, dtype, num_iterations=100, warmup_iterations=10):
    # Ensure TensorFlow is using the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        raise RuntimeError("No GPU devices found.")

    # Generate random data
    A = tf.random.uniform((N, N), dtype=dtype)
    B = tf.random.uniform((N, N), dtype=dtype)

    # Warm-up iterations
    for _ in range(warmup_iterations):
        # Force execution and synchronization
        tf.matmul(A, B).numpy()

    # Benchmark iterations
    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()  # Use perf_counter for more accurate timing
        # Force execution and synchronization
        tf.matmul(A, B).numpy()
        times.append(time.perf_counter() - start_time)  # Use perf_counter

    # Calculate average time and FLOPs
    avg_time = np.mean(times)
    flops = 2 * N**3 / avg_time  # 2*N^3 FLOPs for matrix multiplication
    pflops_per_s = flops / 1e15  # Convert to PFLOPs/s

    print(f"Dtype: {dtype}, Size: {N}x{N}, Avg Time: {avg_time:.5f}s, PFLOPs/s: {pflops_per_s:.5f}")

if __name__ == "__main__":
    N = 4096  # Matrix size
    benchmark_tf_matmul(N, tf.float32)
    benchmark_tf_matmul(N, tf.float16)
