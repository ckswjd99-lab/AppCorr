import torch
from einops import rearrange

class HierarchicalToken:
    def __init__(
        self, 
        lowres_tokens: torch.Tensor, 
        highres_tokens: torch.Tensor, 
        num_pretokens: int = 0
    ):
        self.num_pretokens = num_pretokens
        self.pretokens = lowres_tokens[..., :num_pretokens, :]       # Tensor[..., num_pretokens, D]
        self.lowres_tokens = lowres_tokens[..., num_pretokens:, :]    # Tensor[..., H_low*W_low, D]
        self.highres_tokens = highres_tokens[..., num_pretokens:, :]  # Tensor[..., H_high*W_high, D]

        self.lowres_alive = torch.ones(self.lowres_tokens.shape[-2], dtype=torch.bool, device=lowres_tokens.device)
        self.highres_alive = torch.zeros(self.highres_tokens.shape[-2], dtype=torch.bool, device=highres_tokens.device)

        self.H_low = int((self.lowres_tokens.shape[-2]) ** 0.5)
        self.W_low = self.H_low

        self.H_high = int((self.highres_tokens.shape[-2]) ** 0.5)
        self.W_high = self.H_high

        self.spatial_mapping = self._build_spatial_mapping()        # Tensor[(H_low*W_low), (s1*s2)]
    
    def _build_spatial_mapping(self):
        device = self.lowres_tokens.device
        
        high_grid = torch.arange(
            self.H_high * self.W_high, device=device
        ).reshape(self.H_high, self.W_high)
        
        mapping = rearrange(
            high_grid, 
            '(h s1) (w s2) -> (h w) (s1 s2)', 
            h=self.H_low, w=self.W_low
        )   # Tensor[(H_low*W_low), (s1*s2)]
        
        return mapping

    def get_highres_indices(self, lowres_indices: torch.Tensor):
        mapped = self.spatial_mapping[lowres_indices].reshape(-1).sort().values
        return mapped

    def split_tokens(self, lowres_indices: torch.Tensor):
        highres_indices = self.get_highres_indices(lowres_indices)

        self.lowres_alive[lowres_indices] = False
        self.highres_alive[highres_indices] = True

    def to_tensor(self):
        pretokens = self.pretokens
        lowres_tokens = self.lowres_tokens[..., self.lowres_alive, :]
        highres_tokens = self.highres_tokens[..., self.highres_alive, :]

        return torch.cat([pretokens, lowres_tokens, highres_tokens], dim=-2)

    def from_tensor(self, tokens: torch.Tensor):
        tokens = tokens.to(dtype=self.lowres_tokens.dtype)

        num_pretokens = self.pretokens.shape[-2]
        num_lowres_alive = self.lowres_alive.long().sum().item()
        num_highres_alive = self.highres_alive.long().sum().item()

        self.pretokens = tokens[..., :num_pretokens, :]
        self.lowres_tokens[..., self.lowres_alive, :] = tokens[..., num_pretokens:num_pretokens + num_lowres_alive, :]
        self.highres_tokens[..., self.highres_alive, :] = tokens[..., num_pretokens + num_lowres_alive:, :]

        return self

    def alive_highres_tokens(self):
        return self.highres_tokens[..., self.highres_alive, :]
    
    def clone(self):
        new_instance = HierarchicalToken(
            lowres_tokens=self.lowres_tokens.clone(),
            highres_tokens=self.highres_tokens.clone(),
            num_pretokens=0
        )
        new_instance.num_pretokens = self.num_pretokens
        new_instance.pretokens = self.pretokens.clone()
        new_instance.lowres_alive = self.lowres_alive.clone()
        new_instance.highres_alive = self.highres_alive.clone()

        return new_instance

    def num_tokens(self):
        return self.pretokens.shape[-2] + self.lowres_alive.long().sum().item() + self.highres_alive.long().sum().item()

    def __str__(self):
        return f"HierarchicalToken(num_pretokens={self.num_pretokens}, lowres_alive={self.lowres_alive.sum().item()}/{self.lowres_alive.shape[0]}, highres_alive={self.highres_alive.sum().item()}/{self.highres_alive.shape[0]})"