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
        a = tf.random.uniform((N, N), dtype=tf.bfloat16)
        b = tf.random.uniform((N, N), dtype=tf.bfloat16)

        flop = N * N * 2 * N
        print(f"{flop / 1e9:.2f} GFLOPs")

        results = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        warm_up_iterations = 200
        total_iterations = 500

    for i in range(total_iterations):
        st = time.monotonic()
        c = tf.matmul(a, b)
        et = time.monotonic()
        s = et - st

        if i >= warm_up_iterations and s > 0:
            flops_per_s = flop / s * 1e-12
            results = results.write(i - warm_up_iterations, flops_per_s)
            print(f"{flops_per_s:.2f} TFLOPs/s")
        elif i < warm_up_iterations:
            print(f"Skipping warm-up iteration {i}")

    mean = tf.reduce_mean(results.stack())
    print(f"Mean: {mean:.2f} TFLOPs/s")
