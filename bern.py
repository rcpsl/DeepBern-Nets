from argparse import Namespace
from functools import partial
import jax
import jax.numpy as jnp
from jax import value_and_grad
import optax

import numpy as np

from typing import NamedTuple


class Params(NamedTuple):
    weight: jnp.ndarray


#   bias: jnp.ndarray


# def bern_2D()


@partial(jax.jit, static_argnums=1)
def bern_net_forward(params: Params, deg, x: jnp.ndarray):
    indices = jnp.arange(deg + 1, dtype=jnp.float32).reshape((-1, 1))
    degree = jnp.array(deg, dtype=jnp.float32).reshape((-1, 1))
    basis = bern_basis(x, degree, indices)  # N * dims * coeffs
    out = basis * params.weight.T  # coeffs * dims
    out = jnp.sum(out, axis=-1)
    out = jnp.prod(out, axis=-1).reshape(-1, 1)
    # out = jnp.prod(out, axis())
    return out


def init(rng, deg, dims=1) -> Params:
    """Returns the initial model params."""
    weights_key, _ = jax.random.split(rng)
    weight = jax.random.normal(weights_key, (deg + 1, dims)) * 5 - 2.5
    return Params(weight)


@partial(jax.jit, static_argnums=1)
def loss_fn(params: Params, deg, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the least squares error of the model's predictions on x against y."""
    pred = bern_net_forward(params, deg, x)
    return jnp.mean((pred - y) ** 2)


LEARNING_RATE = 5e-8


@partial(jax.jit, static_argnums=1)
def update(params: Params, deg: int, x: jnp.ndarray, y: jnp.ndarray) -> Params:
    """Performs one SGD update step on params using the given data."""

    loss, grad = value_and_grad(loss_fn)(params, deg, x, y)
    new_params = jax.tree_map(lambda param, g: param - g * LEARNING_RATE, params, grad)

    return new_params, loss, grad


def train(params: Params, deg: int, dataset, epochs, optimizer):
    opt_state = optimizer.init(params)

    @partial(jax.jit, static_argnums=1)
    def step(params, deg, batch, labels, opt_state):
        loss_val, grads = value_and_grad(loss_fn)(params, deg, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    loss_per_epoch = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, y) in enumerate(dataset):
            x = jnp.array(x)
            y = jnp.array(y)
            params, opt_state, batch_loss = step(params, deg, x, y, opt_state)
            epoch_loss += batch_loss

        loss_per_epoch.append(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch + 1} loss = {epoch_loss}")

    return params, loss_per_epoch


@jax.jit
def binom(n, k):
    nCk = jax.lax.lgamma(n + 1) - jax.lax.lgamma(k + 1) - jax.lax.lgamma(n - k + 1)
    nCk = jax.lax.exp(nCk)
    return nCk


@jax.jit
def bern_basis(x, deg, i):
    nCk = binom(deg, i)
    basis = nCk * x**i * (1 - x) ** (deg - i)
    return basis


bern_basis = jax.jit(jax.vmap(bern_basis, (None, None, 0), -1))


def bern_poly(x, coeffs):
    deg = len(coeffs) - 1
    i = jnp.arange(deg + 1, dtype=jnp.float32).reshape((-1, 1))
    deg = jnp.array([deg], dtype=jnp.float32).reshape((-1, 1))
    basis = bern_basis(x, deg, i).squeeze()
    poly = basis @ coeffs
    return poly


def poly_2_bern(i, deg, coeffs):
    j = jnp.arange(i + 1, dtype=jnp.float32)
    iCj = binom(i, j)
    dCj = binom(deg, j)
    b_i = iCj / dCj
    b_i = jnp.vdot(b_i, coeffs[j.astype(jnp.int32)])

    return b_i


def poly_2_bern_all(deg, coeffs):
    result = []
    for i in range(int(deg) + 1):
        idx = float(i)
        result.append(poly_2_bern(idx, deg, coeffs))
    return jnp.array(result)


if __name__ == "__main__":
    x = jnp.array([0, 0, 2, 2], dtype=jnp.float32).reshape((-1, 1))
    deg = jnp.array([3] * 4, dtype=jnp.float32).reshape((-1, 1))
    i = jnp.arange(4, dtype=jnp.float32).reshape((-1, 1))
    y = bern_basis_degree(x, deg, i).block_until_ready()
