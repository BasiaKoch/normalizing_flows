from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare_output_path(out_path: Union[str, Path]) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, out_path: Union[str, Path], *, save_png: bool = True) -> None:
    path = _prepare_output_path(out_path)
    fig.savefig(path, bbox_inches="tight")
    if save_png:
        fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")


def plot_data_splits(
    *,
    train_x: ArrayLike,
    train_class: ArrayLike,
    val_x: ArrayLike,
    val_class: ArrayLike,
    test_x: ArrayLike,
    test_class: ArrayLike,
    out_path: Union[str, Path],
    save_png: bool = True,
) -> None:
    """Visualize train/val/test geometry and class labels."""
    train_x_np = _as_numpy(train_x)
    val_x_np = _as_numpy(val_x)
    test_x_np = _as_numpy(test_x)

    train_c_np = _as_numpy(train_class)
    val_c_np = _as_numpy(val_class)
    test_c_np = _as_numpy(test_class)

    all_x = np.concatenate([train_x_np, val_x_np, test_x_np], axis=0)
    min_xy = all_x.min(axis=0) - 0.3
    max_xy = all_x.max(axis=0) + 0.3

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    splits = [
        ("train", train_x_np, train_c_np),
        ("val", val_x_np, val_c_np),
        ("test", test_x_np, test_c_np),
    ]

    color_handle = None
    for ax, (name, x_np, c_np) in zip(axes, splits):
        color_handle = ax.scatter(
            x_np[:, 0],
            x_np[:, 1],
            c=c_np,
            cmap="coolwarm",
            s=10,
            alpha=0.85,
            edgecolors="none",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{name} split (n={len(x_np)})")
        ax.set_xlabel("x1")
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("x2")
    if color_handle is not None:
        fig.colorbar(color_handle, ax=axes, shrink=0.82, label="class")

    fig.suptitle("Data Exploration: Moons Splits")
    _save_figure(fig, out_path, save_png=save_png)
    plt.close(fig)


def plot_logprob_landscape(
    *,
    flow,
    reference_x: torch.Tensor,
    out_path: Union[str, Path],
    grid_size: int = 220,
    save_png: bool = True,
) -> None:
    """Plot contour map of log p(x) induced by the current flow."""
    x_np = _as_numpy(reference_x)
    lo = np.quantile(x_np, 0.01, axis=0) - 0.35
    hi = np.quantile(x_np, 0.99, axis=0) + 0.35

    gx = np.linspace(lo[0], hi[0], grid_size)
    gy = np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid_np = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    grid = torch.from_numpy(grid_np).to(dtype=reference_x.dtype, device=reference_x.device)
    with torch.no_grad():
        logp = flow.log_prob(grid).detach().cpu().numpy().reshape(grid_size, grid_size)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    contour = ax.contourf(xx, yy, logp, levels=35, cmap="viridis")
    ax.contour(xx, yy, logp, levels=12, colors="white", linewidths=0.35, alpha=0.55)
    ax.scatter(x_np[:, 0], x_np[:, 1], s=4, c="white", alpha=0.25, edgecolors="none")
    ax.set_title("Untrained Flow: log p(x) Landscape")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="log p(x)")

    fig.tight_layout()
    _save_figure(fig, out_path, save_png=save_png)
    plt.close(fig)


def plot_correctness_figure(
    *,
    reconstruction_abs: torch.Tensor,
    invertibility_max_abs_error: float,
    analytic_inv_logdet_value: float,
    finite_diff_logabsdet: float,
    logdet_finite_diff_abs_error: float,
    out_path: Union[str, Path],
    save_png: bool = True,
) -> None:
    """Create the required two-panel correctness diagnostic figure (Figure1c)."""
    pointwise_recon_err = reconstruction_abs.max(dim=1).values.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(pointwise_recon_err, bins=40, color="tab:blue", alpha=0.82, edgecolor="white")
    axes[0].axvline(invertibility_max_abs_error, color="tab:red", linestyle="--", linewidth=2)
    axes[0].set_title("Invertibility Check")
    axes[0].set_xlabel("Per-point max |x_hat - x|")
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.98,
        0.95,
        f"max error = {invertibility_max_abs_error:.2e}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )

    axes[1].bar(
        ["analytic", "finite diff"],
        [analytic_inv_logdet_value, finite_diff_logabsdet],
        color=["tab:green", "tab:orange"],
    )
    axes[1].set_title("Inverse Log-Det Check")
    axes[1].set_ylabel("log |det J_{f^{-1}}|")
    axes[1].text(
        0.5,
        0.95,
        f"abs error = {logdet_finite_diff_abs_error:.2e}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
    )

    fig.tight_layout()
    _save_figure(fig, out_path, save_png=save_png)
    plt.close(fig)


def plot_roundtrip_overlay(
    *,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    out_path: Union[str, Path],
    max_points: int = 400,
    seed: int = 42,
    save_png: bool = True,
) -> None:
    """Overlay original vs reconstructed points and show residual distribution."""
    x_np = _as_numpy(x)
    xhat_np = _as_numpy(x_hat)

    n = x_np.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x_np = x_np[idx]
        xhat_np = xhat_np[idx]

    residual = np.max(np.abs(xhat_np - x_np), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x_np[:, 0], x_np[:, 1], s=12, alpha=0.75, c="tab:blue", label="x")
    axes[0].scatter(xhat_np[:, 0], xhat_np[:, 1], s=12, alpha=0.75, c="tab:orange", marker="x", label="x_hat")
    axes[0].set_title("Roundtrip Overlay")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", frameon=False)

    axes[1].hist(residual, bins=35, color="tab:purple", alpha=0.82, edgecolor="white")
    axes[1].set_title("Roundtrip Error (Sampled)")
    axes[1].set_xlabel("max |x_hat - x|")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    _save_figure(fig, out_path, save_png=save_png)
    plt.close(fig)
