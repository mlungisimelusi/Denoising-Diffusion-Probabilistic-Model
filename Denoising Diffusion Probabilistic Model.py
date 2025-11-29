import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
])

dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)

T = 200  # number of diffusion steps
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x, t):
        return self.net(x)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(5):
    for x, _ in dataloader:
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=x.device).long()
        x_t, noise = forward_diffusion_sample(x, t)
        pred_noise = model(x_t, t)
        loss = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

@torch.no_grad()
def sample(model, n=16):
    x = torch.randn((n, 1, 28, 28)).to(device)
    for t in reversed(range(T)):
        t_tensor = torch.full((n,), t, device=x.device, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else 0
        beta = betas[t]
        alpha = alphas[t]
        alpha_hat_t = alpha_hat[t]
        pred_noise = model(x, t_tensor)
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) + torch.sqrt(beta) * z
    return x

samples = sample(model)
grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()

