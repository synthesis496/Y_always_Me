#Copyright(c)2026 chutiphong bunloed
#All Rights Reserved.
import torch
import torch.nn as nn


class DNASwingingBranchless(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.register_buffer("bond_weights", torch.tensor([2.0, 3.0]))
        self.register_buffer("PHI", torch.tensor(1.6180339887498949))
        self.register_buffer("GOLDEN_FREQ", torch.tensor(34.0 / 21.0))

    def tent_map_step(self, key_state: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.abs(2.0 * key_state - 1.0)

    def forward(self, state: torch.Tensor, key_seed: torch.Tensor, t_step: int):
        next_key = self.tent_map_step(key_seed)

        half_dim = self.d_model // 2
        strand_a, strand_b = state.chunk(2, dim=-1)

        swing_factor = torch.where(
            next_key[..., :half_dim] > 0.5,
            self.bond_weights[0],
            self.bond_weights[1],
        )

        swung_a = strand_b * swing_factor
        swung_b = strand_a * (5.0 - swing_factor)
        swung_state = torch.cat([swung_a, swung_b], dim=-1)

        t_tensor = torch.as_tensor(t_step, device=state.device)
        is_singularity = torch.eq((t_tensor + 1) % 16, 0)

        singularity_val = torch.sin(-swung_state * self.PHI)
        normal_val = torch.cos(swung_state * self.GOLDEN_FREQ)

        swung_state = torch.where(is_singularity, singularity_val, normal_val)

        return swung_state, next_key


if __name__ == "__main__":
    torch.manual_seed(42)
    layer = DNASwingingBranchless(d_model=512)
    state = torch.randn(2, 512)
    key = torch.rand(2, 512)

    for t in range(32):
        state, key = layer(state, key, t_step=t)
        if (t + 1) % 8 == 0:
            print(f"step {t+1:2d} | state std={state.std():.4f} | key mean={key.mean():.4f}")
