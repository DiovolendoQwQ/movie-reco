import implicit
try:
    print(f"Implicit CUDA available: {implicit.gpu.HAS_CUDA}")
except AttributeError:
    print("Could not directly check implicit.gpu.HAS_CUDA. Trying PyTorch check as an alternative.")
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices found: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError:
        print("PyTorch not installed, cannot check CUDA availability via PyTorch.")

exit()