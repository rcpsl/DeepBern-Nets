import torch
import torch.nn as nn


class BernsteinLayer(nn.Module):
    def __init__(self, in_shape, degree: int):
        super().__init__()
        self.degree = degree
        self.in_shape = in_shape
        _basis_indices_tensor = torch.arange(degree + 1).reshape((-1, degree + 1))
        _deg_tensor = torch.tensor([degree]).reshape(-1, 1)
        nCk_tensor = self.binom(_deg_tensor, _basis_indices_tensor)
        input_bounds = torch.zeros((*in_shape, 2))
        self.register_buffer("input_bounds", input_bounds)
        self.register_buffer("_basis_indices", _basis_indices_tensor)
        self.register_buffer("_deg_tensor", _deg_tensor)
        self.register_buffer("nCk", nCk_tensor)
        bern_coeffs = torch.ones(*in_shape, degree + 1)
        init_std = torch.ones_like(bern_coeffs) / torch.tensor(in_shape).prod()
        self.bern_coeffs = nn.Parameter(
            bern_coeffs * torch.normal(torch.zeros_like(bern_coeffs), init_std)
        )

    @property
    def bern_bounds(self):
        lb, _ = self.bern_coeffs.min(axis=-1, keepdim=True)
        ub, _ = self.bern_coeffs.max(axis=-1, keepdim=True)
        return torch.concat((lb, ub), dim=-1)

    def subinterval_bounds(self, bounds):
        """Compute Bernstein coeffs for interval [alpha,beta]"""
        in_bounds = self.input_bounds.unsqueeze(0)
        bounds = (bounds - in_bounds[..., 0:1]) / (
            in_bounds[..., 1:] - in_bounds[..., 0:1]
        )
        alpha = bounds[..., 0].unsqueeze(-1)
        beta = bounds[..., 1].unsqueeze(-1)
        zero_to_beta_coeffs = torch.zeros(
            bounds.shape[0],
            *self.in_shape,
            self.degree + 1,
            self.degree + 2,
            device=alpha.device,
        )  # Pad with an extra row and col
        zero_to_beta_coeffs[..., 0, :-1] = self.bern_coeffs
        # temp = self.bern_coeffs
        for i in range(1, zero_to_beta_coeffs.shape[-2]):
            zero_to_beta_coeffs[..., i, 1:] = (1 - beta) * zero_to_beta_coeffs[
                ..., i - 1, :-1
            ].clone() + beta * zero_to_beta_coeffs[..., i - 1, 1:].clone()

        zero_to_beta_coeffs = zero_to_beta_coeffs[..., :-1].diagonal(dim1=-2, dim2=-1)

        gamma = alpha / beta
        alpha_to_beta_coeffs = torch.zeros(
            bounds.shape[0],
            *self.in_shape,
            self.degree + 1,
            self.degree + 2,
            device=alpha.device,
        )  # Pad with an extra row and col
        alpha_to_beta_coeffs[..., 0, :-1] = zero_to_beta_coeffs
        for i in range(1, alpha_to_beta_coeffs.shape[-2]):
            alpha_to_beta_coeffs[..., i, 1:] = (1 - gamma) * alpha_to_beta_coeffs[
                ..., i - 1, :-1
            ].clone() + gamma * alpha_to_beta_coeffs[..., i - 1, 1:].clone()

        new_coeffs_lb_ub = alpha_to_beta_coeffs[..., -2]
        lb = new_coeffs_lb_ub.min(axis=-1, keepdim=True)[0]
        ub = new_coeffs_lb_ub.max(axis=-1, keepdim=True)[0]
        return torch.concat((lb, ub), -1)

    def binom(self, n, k):
        nCk = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        nCk = torch.exp(nCk)
        nCk = torch.floor(nCk + 0.5)
        return nCk

    def bern_basis(self, x):
        y = x.unsqueeze(-1)
        basis = (
            self.nCk
            * (y) ** self._basis_indices
            * (1 - y) ** (self._deg_tensor - self._basis_indices)
        )
        return basis
        # return basis / diff

    def forward(self, x):
        x = (x - self.input_bounds[..., 0]) / (
            self.input_bounds[..., 1] - self.input_bounds[..., 0]
        )
        basis = self.bern_basis(x)
        with torch.no_grad():
            basis_shape = torch.tensor(basis.shape)
            basis_sum_to_one = torch.isclose(
                torch.sum(basis, axis=-1).sum(), basis_shape[:-1].prod().float()
            )
            if not basis_sum_to_one:
                raise Exception(
                    f"Basis doesn't sum to 1, {torch.sum(basis,axis = -1).sum()}"
                )
        out = basis * self.bern_coeffs
        out = out.sum(axis=-1)
        return out
