"""Standalone latent-space regularizers."""

from typing import Optional

import jax
import jax.numpy as jnp


def _epps_pulley_grid(
    num_points: int = 17,
    t_max: float = 3.0,
    dtype=jnp.float32,
):
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError(f"num_points must be an odd integer >= 3, got {num_points}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    t = jnp.linspace(0.0, t_max, num_points, dtype=dtype)
    dt = jnp.asarray(t_max / (num_points - 1), dtype=dtype)
    base_weights = jnp.full((num_points,), 2.0 * dt, dtype=dtype)
    base_weights = base_weights.at[0].set(dt)
    base_weights = base_weights.at[-1].set(dt)
    weights = base_weights * jnp.exp(-0.5 * jnp.square(t))
    return t, weights


def epps_pulley_statistic(
    projected: jax.Array,
    num_points: int = 17,
    t_max: float = 3.0,
):
    """Compute one Epps-Pulley statistic per projected slice."""
    if projected.ndim < 2:
        raise ValueError(
            "projected must have shape (..., num_samples, num_slices), "
            f"got {projected.shape}"
        )

    projected = jnp.asarray(projected)
    num_samples = projected.shape[-2]
    t, weights = _epps_pulley_grid(
        num_points=num_points,
        t_max=t_max,
        dtype=projected.dtype,
    )
    gaussian_cf = jnp.exp(-0.5 * jnp.square(t))

    xt = projected[..., :, :, None] * t
    cos_mean = jnp.mean(jnp.cos(xt), axis=-3)
    sin_mean = jnp.mean(jnp.sin(xt), axis=-3)

    err = jnp.square(cos_mean - gaussian_cf) + jnp.square(sin_mean)
    return jnp.tensordot(err, weights, axes=([-1], [0])) * num_samples


def sample_unit_gaussian_slices(
    rng: jax.Array,
    dim: int,
    num_slices: int,
    dtype=jnp.float32,
):
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")

    directions = jax.random.normal(rng, (dim, num_slices), dtype=dtype)
    norms = jnp.linalg.norm(directions, axis=0, keepdims=True)
    norms = jnp.maximum(norms, jnp.asarray(1e-8, dtype=dtype))
    return directions / norms


def sliced_epps_pulley_loss(
    embeddings: jax.Array,
    rng: jax.Array,
    num_slices: int = 32,
    num_points: int = 17,
    t_max: float = 3.0,
    clip_value: Optional[float] = None,
    reduction: str = "mean",
):
    """Aggregate sliced Epps-Pulley statistics into a scalar loss.

    The classical Epps-Pulley statistic scales linearly with the number of
    samples. For optimization we normalize that factor back out so the loss
    scale is much less sensitive to PPO minibatch size.
    """
    if embeddings.ndim < 2:
        raise ValueError(
            f"embeddings must have shape (..., num_samples, dim), got {embeddings.shape}"
        )

    embeddings = jnp.asarray(embeddings)
    directions = sample_unit_gaussian_slices(
        rng,
        dim=embeddings.shape[-1],
        num_slices=num_slices,
        dtype=embeddings.dtype,
    )
    projected = jnp.einsum("...nd,dk->...nk", embeddings, directions)
    stats = epps_pulley_statistic(projected, num_points=num_points, t_max=t_max)
    sample_count = jnp.asarray(embeddings.shape[-2], dtype=stats.dtype)
    stats = stats / jnp.maximum(sample_count, jnp.asarray(1.0, dtype=stats.dtype))

    if clip_value is not None:
        clip_value = jnp.asarray(clip_value, dtype=stats.dtype)
        stats = jnp.where(stats < clip_value, 0.0, stats)

    if reduction == "mean":
        return jnp.mean(stats)
    if reduction == "sum":
        return jnp.sum(stats)
    if reduction in ("none", None):
        return stats
    raise ValueError(f"Unknown reduction {reduction!r}")
