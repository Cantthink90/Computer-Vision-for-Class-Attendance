from alexTest import AlexNet, evaluate, testLoader, num_classes


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "AlexNetTrained.pth"

model = AlexNet(num_classes=num_classes)
state_dict = torch.load(PATH)

# Remap torchvision AlexNet keys to your custom AlexNet keys
key_map = {"features": "convolutional", "classifier": "linear"}
new_state_dict = {}
for k, v in state_dict.items():
    for old, new in key_map.items():
        if k.startswith(old):
            k = k.replace(old, new, 1)
            break
    new_state_dict[k] = v

# Drop the final layer keys since class counts don't match
new_state_dict.pop("linear.6.weight", None)
new_state_dict.pop("linear.6.bias", None)

model.load_state_dict(new_state_dict, strict=False)

model.to(device)
model.eval()


evaluate(model, testLoader, "Test Before SVD")


def svd_approx(kernel):
    original_shape = kernel.shape
    k2d = kernel.reshape(original_shape[0], -1)
    U, S, Vh = torch.linalg.svd(k2d, full_matrices=False)
    rank = round(S.numel() * 0.5)
    print(f"full rank: {S.numel()} | kept rank: {rank}")
    k2d_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    return k2d_approx.reshape(original_shape)


with torch.no_grad():
    for layer in list(model.convolutional) + list(model.linear):
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.copy_(svd_approx(layer.weight.data.clone()))

evaluate(model, testLoader, "Test After SVD")
