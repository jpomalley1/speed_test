import pandas as pd
import time
import tensorflow as tf
import random

print("TensorFlow version:", tf.__version__)
N = 4096
random.seed(777)

### speed test

if __name__ == "__main__":
    # Enable GPU acceleration
    with tf.device('/GPU:0'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # N^2
        a = tf.random.uniform((N, N), dtype=tf.float32)  # Specify dtype here if needed
        b = tf.random.uniform((N, N), dtype=tf.float32)  # Specify dtype here if needed

        print(f"Data type of matrix A: {a.dtype}")
        print(f"Data type of matrix B: {b.dtype}")

        flop = N * N * 2 * N
        print(f"{flop / 1e9:.2f} GFLOPs")

        results = []

        warm_up_iterations = 200
        total_iterations = 500

        for i in range(total_iterations):
            st = time.monotonic()
            c = tf.matmul(a, b)
            et = time.monotonic()
            s = et - st

            if i >= warm_up_iterations and s > 0:
                flops_per_s = flop / s * 1e-15  # Convert to PFLOPs/s
                results.append(flops_per_s)
                print(f"{flops_per_s:.2f} PFLOPs/s")
            elif i < warm_up_iterations:
                print(f"Skipping warm-up iteration {i}")

        mean_flops_per_s = sum(results) / len(results)
        print(f"Mean: {mean_flops_per_s:.2f} PFLOPs/s")

