import torch
import numpy as np
import matplotlib.pyplot as plt

def process_ikeda(a):
    """Display an array of iteration counts as a
    colorful picture of the Ikeda map fractal."""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                          30 + 50 * np.sin(a_cyclic),
                          155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers
Y, X = np.mgrid[-1.5:1.5:0.002, -1.5:1.5:0.002]

# Load into PyTorch tensors
x = torch.Tensor(X).to(device)
y = torch.Tensor(Y).to(device)
z = x + 1j * y  # Use PyTorch's complex number data type

zs = z.clone()
ns = torch.zeros_like(x).to(device)

# Parameters for the Ikeda map
u = 0.98
iterations = 200

for i in range(iterations):
    # Apply the Ikeda map equation
    t = 0.4 - 6 / (1 + zs.real**2 + zs.imag**2)
    x_new = 1 + u * (zs.real * torch.cos(t) - zs.imag * torch.sin(t))
    y_new = u * (zs.real * torch.sin(t) + zs.imag * torch.cos(t))
    zs = x_new + 1j * y_new
    
    # Update variables to compute
    ns += torch.abs(zs) < 2.0

# Convert the result to a NumPy array for visualization
fractal_image = process_ikeda(ns.cpu().numpy())

# Plot
fig = plt.figure(figsize=(16, 10))
plt.imshow(fractal_image)
plt.tight_layout(pad=0)
plt.show()