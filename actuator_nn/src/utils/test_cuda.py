import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Try to create a CUDA tensor
try:
    x = torch.cuda.FloatTensor(1)
    print("Successfully created a CUDA tensor")
except Exception as e:
    print(f"Error creating CUDA tensor: {e}")

# Try to move a tensor to GPU
try:
    x = torch.FloatTensor(1)
    x = x.cuda()
    print("Successfully moved tensor to GPU")
except Exception as e:
    print(f"Error moving tensor to GPU: {e}")