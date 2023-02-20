import os

from tensorflow.config import experimental, list_physical_devices
from tensorflow.keras import mixed_precision

"""
Install tensorflow gpu support as win native. But note that this is not compatible with 
the most recent PyTorch version and thus won't work with YOLOv8:
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Or better
conda install -c conda-forge cudatoolkit-dev cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
"""


def init_gpu_tpu():
    """
    This function initializes the GPU/TPU and sets the global mixed precision policy. If TPUs
    are detected, it sets the policy to 'mixed_bfloat16' and if GPUs are detected, it sets the
    policy to 'mixed_float16' and selects the first GPU for usage. If no TPUs or GPUs are
    detected, it prints a message indicating so.
    """
    # Use mixed precision policy if TPU/GPUs are detected for additional performance boost
    # see https://www.tensorflow.org/guide/mixed_precision
    if experimental.list_physical_devices(
        "TPU"
    ):  # Googles Tensor Processing Units
        print(
            "Available TPUs: ", len(experimental.list_physical_devices("TPU"))
        )
        mixed_precision.set_global_policy(
            mixed_precision.Policy("mixed_bfloat16")
        )
    elif experimental.list_physical_devices("GPU"):  # NVidia GPU
        print(
            "Available GPUs: ", len(experimental.list_physical_devices("GPU"))
        )
        print(list_physical_devices("GPU"))
        print("Selecting the first GPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the first GPU
        mixed_precision.set_global_policy(
            mixed_precision.Policy("mixed_float16")
        )
    else:
        print("No TPUs or GPUs detected.")
    # Note: if mixed precision float16 is used, your output layer should be transferred to
    # float32 because float16 are not always numerically stable.
