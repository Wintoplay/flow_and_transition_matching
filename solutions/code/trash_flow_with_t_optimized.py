# Lab 3: A Conditional Generative Model for Images
# In this lab, we will study conditional generation on images from the MNIST dataset.
# We will use Classifier-Free Guidance (CFG) and a U-Net architecture.

from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
degree_of_freedom = 30
t_distribution = torch.distributions.studentT.StudentT(degree_of_freedom)

# ### Part 0: Recycling Components from Previous Labs
# We re-import components from previous labs, with some important updates.

# Old `Sampleable` class from labs one and two.
class OldSampleable(ABC):
    """
    Distribution which can be sampled from
    """        
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
        """
        pass

# We generalize `Sampleable` to accommodate labels as well.
# The sample method will now return both samples and optional labels.
class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """ 
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass

# For distributions like a Gaussian, labels are optional and can be None.
class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
        
    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * t_distribution.sample((num_samples, *self.shape)).to(self.dummy.device), None

# Updating `ConditionalProbabilityPath` to handle labels and image shapes (batch_size, c, h, w).
# The time variable `t` is maintained in the shape (batch_size, 1, 1, 1) to avoid broadcasting issues.
class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        pass

# We re-introduce `GaussianConditionalProbabilityPath` with `Alpha` and `Beta` schedules.
# To avoid broadcasting issues with image tensors, alpha(t) and beta(t) are reshaped to (batch_size, 1, 1, 1).
class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """ 
        t_flat = t.view(-1, 1)
        dt = vmap(jacrev(self))(t_flat)
        return dt.view_as(t)
    
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """ 
        t_flat = t.view(-1, 1)
        dt = vmap(jacrev(self))(t_flat)
        return dt.view_as(t)

class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """ 
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """ 
        return torch.ones_like(t)
        
class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return 1-t
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """ 
        return - torch.ones_like(t)
    
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * t_distribution.sample(z.shape).to(z.device)
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        # Ensure no division by zero for beta_t at t=1
        beta_t = torch.where(beta_t == 0, torch.full_like(beta_t, 1e-9), beta_t)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2

# Updating ODE, SDE, and Simulator classes for image shapes and optional conditioning inputs.
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1), desc="Simulating ODE"):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt,t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

# Adding back the definition of `Trainer`.
MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs), desc="Training")
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(device=device, **kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {epoch}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()

# # Part 1: Getting a Feel for MNIST
# We'll load MNIST and experiment with adding noise using `ConditionalGaussianProbabilityPath`.
class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        # It's better to have a persistent dataloader
        self.dataset = datasets.MNIST(
            root='/home/win/big_dataset',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        # Create a dataloader to efficiently sample a batch
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=num_samples, shuffle=True)
        samples, labels = next(iter(dataloader))
        return samples.to(self.dummy.device), labels.to(self.dummy.device)

# Visualize samples along the conditional probability path from data to noise.
# Change these!
num_rows = 3
num_cols = 3
num_timesteps = 5

# Initialize our sampler
sampler = MNISTSampler().to(device)

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = sampler,
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Sample 
num_samples = num_rows * num_cols
z, _ = path.p_data.sample(num_samples)
z = z.view(-1, 1, 32, 32).to(device)

# Setup plot
fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_timesteps, 6))

# Sample from conditional probability paths and graph
ts = torch.linspace(0, 1, num_timesteps).to(device)
for tidx, t in enumerate(ts):
    tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1) # (num_samples, 1, 1, 1)
    xt = path.sample_conditional_path(z, tt) # (num_samples, 1, 32, 32)
    grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
    axes[tidx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axes[tidx].set_title(f't = {t.item():.2f}')
    axes[tidx].axis("off")
plt.tight_layout()
plt.show()

# # Part 2: Classifier Free Guidance
# ## Problem 2.1: Classifier Free Guidance
# We want to condition our generation on a specific class label (e.g., "generate an 8").
# This is done by learning a conditional vector field u_t(x|y).
# To improve perceptual quality, we use Classifier-Free Guidance (CFG).

# The CFG guided vector field is a linear combination of a conditional and an unconditional vector field:
# u_tilde(x|y) = (1-w) * u(x|∅) + w * u(x|y)
# where ∅ is a null label representing the absence of conditioning, and w is the guidance scale.

# To train a single model for both conditional and unconditional generation, we replace the true label y
# with a null label (e.g., 10) with some probability η during training.
# The training objective is:
# 1. Sample an image z and label y from the dataset.
# 2. With probability η, replace y with the null label ∅.
# 3. Sample a time t from U[0,1].
# 4. Sample x from the conditional path p_t(x|z).
# 5. Minimize the squared error between the model's predicted vector field u_t^θ(x|y) and the true field u_t^ref(x|z).

# ## Question 2.2: Training for Classifier-Free Guidance
# Implement the training objective.
class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        # Ensure y is on the correct device
        y = y.to(x.device)
        
        # Calculate the guided vector field
        guided_vector_field = self.net(x, t, y)
        
        # Create the null label for the unguided vector field
        unguided_y = torch.full_like(y, 10) # 10 is the null label
        unguided_vector_field = self.net(x, t, unguided_y)
        
        # Combine them using the guidance scale
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

# Your job: Fill in `CFGFlowTrainer.get_train_loss`.
# This class implements the CFG training objective.
class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float):
        assert 0 < eta < 1
        super().__init__(model)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # (bs, c, h, w), (bs,)
        z, y = z.to(device), y.to(device)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        # Create a mask for dropout
        dropout_mask = torch.rand(y.shape[0], device=device) < self.eta
        y[dropout_mask] = 10 # 10 is the null label
        
        # Step 3: Sample t and x
        # The notebook uses a logit-normal distribution for t instead of uniform.
        # This can be a variance reduction technique that focuses training on times near 0 and 1.
        t_raw = torch.randn(batch_size, 1, 1, 1, device=device) # Logit-Normal sampling
        t = torch.sigmoid(t_raw)
        x = self.path.sample_conditional_path(z, t) # (bs, 1, 32, 32)

        # Step 4: Regress and output loss
        ut_theta = self.model(x, t, y) # (bs, 1, 32, 32)
        ut_ref = self.path.conditional_vector_field(x, z, t) # (bs, 1, 32, 32)
        
        # Calculate loss (e.g., Mean Squared Error)
        loss = nn.functional.mse_loss(ut_theta, ut_ref)
        return loss

# # Part 3: An Architecture for Images
# A simple MLP is insufficient for high-dimensional image data.
# We will use a convolutional architecture called the U-Net.

# ## Question 3.1: Building a U-Net
# Below, we implement the U-Net shown in the lab diagram.
class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.GroupNorm(int(channels/4),channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.GroupNorm(int(channels/4),channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )
        # self.norm = nn.GroupNorm(int(channels/4), channels)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone() # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x) # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x) # (bs, c, h, w)

        # Add back residual
        x = x + res # (bs, c, h, w)

        return x
        
class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x

class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
            
        return x
        
class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w) 
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x
        
class MNISTUNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int): 
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)

        # Encoders, Midcoders, and Decoders
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Build Encoder
        for i in range(len(channels) - 1):
            self.encoders.append(Encoder(channels[i], channels[i+1], num_residual_layers, t_embed_dim, y_embed_dim))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        
        # Build Decoder
        for i in reversed(range(len(channels) - 1)):
            self.decoders.append(Decoder(channels[i+1], channels[i], num_residual_layers, t_embed_dim, y_embed_dim))
            
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(32, int(channels[0]/4)), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop() # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x

# # Your job: Pick two components of the architecture above and explain them.
# # 
# # Your answer:

# # Component 1 (FourierEncoder):
# # 1. Role: The `FourierEncoder` is responsible for encoding the continuous time variable `t` into a high-dimensional vector representation. This is crucial because neural networks, particularly transformers and U-Nets in diffusion models, need a way to process the notion of "time" or "noise level" as a condition. A simple scalar `t` is often not expressive enough.
# # 2. Inputs/Outputs: It takes a tensor `t` of shape (batch_size, 1, 1, 1) as input and outputs a tensor of shape (batch_size, `dim`), where `dim` is the specified embedding dimension.
# # 3. How it works: It works by applying a set of sinusoidal functions (sines and cosines) with different frequencies to the input time `t`. The frequencies are determined by a learnable `weights` parameter. This is inspired by positional encodings in transformers. By projecting the single time dimension onto multiple sinusoidal dimensions, it creates a rich, periodic, and non-local representation that helps the model distinguish between different noise levels more effectively.

# # Component 2 (ResidualLayer):
# # 1. Role: The `ResidualLayer` is the fundamental building block of the U-Net's encoder, decoder, and mid-section. Its primary role is to process the image features at a specific resolution while incorporating the time and class-label conditions. The "residual" aspect helps in training very deep networks by preventing the vanishing gradient problem.
# # 2. Inputs/Outputs: It takes the image feature map `x` (shape: bs, c, h, w), the time embedding `t_embed` (shape: bs, t_embed_dim), and the label embedding `y_embed` (shape: bs, y_embed_dim) as input. It outputs a feature map of the same shape as the input `x`.
# # 3. How it works: It first saves the input `x` as a residual connection. The input then passes through two convolutional blocks (often with Group Normalization, an activation function like SiLU, and Dropout). The time and label embeddings are passed through separate linear "adapter" layers to match the number of channels in the image feature map. These adapted embeddings are then added to the feature map, effectively conditioning the convolutional operations on the time and label. Finally, the output of the convolutional blocks is added back to the original input residual `x`, creating the final output. This allows the layer to learn modifications to the identity function, which is easier than learning the entire transformation from scratch.

# ## Question 3.2: Training a U-Net for Classifier-Free Guidance
# Now let's train!
# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 128,
    y_embed_dim = 128
).to(device)

# Initialize trainer
trainer = CFGTrainer(
    path = path,
    model = unet,
    eta = 0.1 # Dropout probability for labels
)

# Train the model
trainer.train(
    num_epochs = 5000, # A higher number of epochs is recommended for better results
    device = device,
    lr = 1e-4,
    batch_size = 250
)

# ## Question 3.3: Generating Images
# Now we use the trained model to generate images for each class.
print("Generating images...")

# Configuration
num_classes = 10
samples_per_class = 10
guidance_scale = 5.0
num_timesteps = 100

# Prepare for generation
ode = CFGVectorFieldODE(net=unet, guidance_scale=guidance_scale)
simulator = EulerSimulator(ode=ode)
all_generated_images = []

# Generate images for each class
for class_label in range(num_classes):
    print(f"Generating for class: {class_label}")
    
    # Prepare labels
    y = torch.full((samples_per_class,), class_label, dtype=torch.long, device=device)
    
    # Sample initial noise from the simple distribution (t=1)
    x0, _ = path.p_simple.sample(samples_per_class)
    x0 = x0.to(device)
    
    # Define timesteps for simulation from t=1 to t=0
    ts = torch.linspace(1, 0, num_timesteps, device=device).view(1, -1, 1, 1, 1).expand(samples_per_class, -1, 1, 1, 1)
    
    # Simulate the ODE
    generated_images = simulator.simulate(x0, ts, y=y)
    all_generated_images.append(generated_images)

# Concatenate and visualize
final_images = torch.cat(all_generated_images, dim=0)
grid = make_grid(final_images, nrow=samples_per_class, normalize=True, value_range=(-1, 1))

# Plotting
plt.figure(figsize=(12, 12))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
plt.title(f"Generated MNIST Digits with CFG (Guidance Scale: {guidance_scale})")
plt.axis("off")
# plt.show()
plt.savefig(f"image_out/t_flow_fixed.png")