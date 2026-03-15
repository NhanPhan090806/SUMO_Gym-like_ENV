import torch
print(torch.cuda.is_available())        # Phải ra True
print(torch.cuda.get_device_name(0))    # Phải hiện "NVIDIA GeForce RTX 5060 Ti"