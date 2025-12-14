import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterNetwork(nn.Module):
    """
    Meta-learnable router network that predicts:
    - sigma: Gaussian blur sigma for edge detection
    - laplacian_kernel: 3x3 Laplacian kernel weights
    - gamma: Edge feature fusion weight

    Input: Image features from CLIP encoder
    Output: Task-adaptive hyperparameters
    """

    def __init__(self, cfg, input_dim=512):
        super(RouterNetwork, self).__init__()

        self.cfg = cfg
        self.input_dim = input_dim
        self.hidden_dim = cfg.TRAINER.BiMC.META.ROUTER_HIDDEN_DIM

        # Feature encoder: aggregate image features into task representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # Sigma predictor: predicts Gaussian blur sigma (0.5 ~ 2.5)
        self.sigma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1], will be scaled to [0.5, 2.5]
        )

        # Laplacian kernel predictor: predicts 3x3 kernel weights
        self.kernel_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 3x3 = 9 weights
        )

        # Gamma predictor: predicts edge fusion weight (0.0 ~ 1.0)
        self.gamma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        with torch.no_grad():
            self.gamma_head[-2].bias.fill_(0.4)

        # Initialize kernel head to output Laplacian-like kernel initially
        self._initialize_kernel_head()

    def _initialize_kernel_head(self):
        """Initialize kernel head to output standard Laplacian kernel"""
        # Standard Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        target_kernel = torch.tensor([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0])

        # Set bias of last layer to target kernel
        with torch.no_grad():
            self.kernel_head[-1].bias.copy_(target_kernel)
            # Small weight initialization
            self.kernel_head[-1].weight.mul_(0.01)

    def forward(self, image_features):
        """
        Args:
            image_features: (B, D) - CLIP encoded image features

        Returns:
            sigma: (B, 1) - Gaussian blur sigma in [0.5, 2.5]
            kernel: (B, 1, 3, 3) - Laplacian kernel weights
            gamma: (B, 1) - Edge fusion weight in [0.0, 1.0]
        """
        # Encode features
        encoded = self.feature_encoder(image_features)

        # Predict sigma: scale [0, 1] -> [0.5, 2.5]
        sigma_normalized = self.sigma_head(encoded)  # (B, 1)
        sigma = 0.5 + sigma_normalized * 2.0  # Scale to [0.5, 2.5]

        # Predict Laplacian kernel: reshape to 3x3
        kernel_flat = self.kernel_head(encoded)  # (B, 9)
        kernel = kernel_flat.view(-1, 1, 3, 3)  # (B, 1, 3, 3)

        # Predict gamma: already in [0, 1] from sigmoid
        gamma = self.gamma_head(encoded)  # (B, 1)

        return sigma, kernel, gamma

    def forward_aggregated(self, image_features):
        """
        Aggregate predictions across batch using mean

        Args:
            image_features: (B, D) - CLIP encoded image features

        Returns:
            sigma: scalar - Mean sigma across batch
            kernel: (1, 1, 3, 3) - Mean kernel across batch
            gamma: scalar - Mean gamma across batch
        """
        sigma, kernel, gamma = self.forward(image_features)

        # Aggregate by taking mean across batch
        sigma_agg = sigma.mean()
        kernel_agg = kernel.mean(dim=0, keepdim=True)  # (1, 1, 3, 3)
        gamma_agg = gamma.mean()

        return sigma_agg, kernel_agg, gamma_agg

    def get_default_params(self):
        """
        Return default hyperparameters (for initialization or fallback)

        Returns:
            sigma: default sigma value (1.0)
            kernel: standard Laplacian kernel
            gamma: default gamma value (0.5)
        """
        device = next(self.parameters()).device

        sigma = torch.tensor(1.0, device=device)
        kernel = torch.tensor([
            [[[0.0, 1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0, 1.0, 0.0]]]
        ], device=device)  # (1, 1, 3, 3)
        gamma = torch.tensor(0.5, device=device)

        return sigma, kernel, gamma
