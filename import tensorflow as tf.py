import tensorflow as tf

# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Assuming you want to use the first (and only) GPU
    gpu_index = 0

    # Check if the chosen GPU index is within the available GPUs
    if gpu_index < len(gpus):
        gpu = gpus[gpu_index]

        # Get the GPU's compute capability
        device_details = tf.config.experimental.get_device_details(gpu)
        compute_capability = device_details['compute_capability']
        major, minor = compute_capability

        # Dictionary of CUDA cores for different compute capabilities
        cuda_cores = {
            (3, 0): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,
            (6, 0): 128,
            (6, 1): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 64
        }

        # Get the CUDA cores for the compute capability or use a default value
        cuda_cores_for_capability = cuda_cores.get((major, minor), 0)

        # Get the clock speed using an alternative method
        clock_speed = tf.config.experimental.get_memory_info(gpu).attributes['clockRate'] / 1e3  # in GHz

        # Number of Fused Multiply-Add (FMA) operations per CUDA core
        fma_ops_per_core = 2  # Common value for modern architectures

        # Calculate theoretical teraflops
        teraflops = cuda_cores_for_capability * clock_speed * fma_ops_per_core / 1e3  # Convert to teraflops

        print(f'Theoretical Teraflops for GPU {gpu_index}: {teraflops:.2f} TFLOPS')  # cSpell:ignore TFLOPS
    else:
        print(f'Invalid GPU index {gpu_index}. Please choose a valid index.')
else:
    print('No GPU available.')

