import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb
import minimal_GMVAE as gmvae


# --------------------------
# Small utilities
# --------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pca_project(X: torch.Tensor, n_components: int = 2):
    """PCA projection via SVD, returns Z, W, mean (CPU tensors)."""
    if X.device.type != "cpu":
        X = X.cpu()
    X = X.float()
    mean = X.mean(dim=0)
    Xc = X - mean.unsqueeze(0)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components].T
    Z = Xc @ W
    return Z, W, mean


def one_hot(y: torch.Tensor, K: int):
    oh = torch.zeros(y.shape[0], K, device=y.device, dtype=torch.float32)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh


# --------------------------
# Plotting helpers -> W&B Images
# --------------------------
def fig_to_wandb_image(fig, caption: str):
    fig.tight_layout()
    img = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return img


@torch.no_grad()
def collect_latent_mu(model, loader, device, max_points=5000):
    model.eval()
    mus, comps, ytrues = [], [], []
    n = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            ytrues.append(y.cpu())
        else:
            x, y = (batch[0] if isinstance(batch, (tuple, list)) else batch), None

        x = x.to(device)
        out = model.encoder(x)
        mu = out["mu_z"].detach().cpu()
        qy = out["q_y"].detach().cpu()
        comp = torch.argmax(qy, dim=-1)

        mus.append(mu)
        comps.append(comp)
        n += x.size(0)
        if n >= max_points:
            break

    mus = torch.cat(mus, dim=0)[:max_points]
    comps = torch.cat(comps, dim=0)[:max_points]
    ytrue = torch.cat(ytrues, dim=0)[:max_points] if len(ytrues) else None
    return mus, comps, ytrue


@torch.no_grad()
def reco_grid_per_label(
    model,
    loader,
    device,
    K: int,
    n_per_label: int = 3,
    class_names=None,
    show_means: bool = False,  # not used here, but kept for symmetry
):
    """
    Collect first n_per_label examples per true label (works with label-ordered val loader),
    run reco, and plot grid with true/pred/conf.
    """
    model.eval()
    xs_by = {k: [] for k in range(K)}

    for batch in loader:
        x, y = batch[0], batch[1]
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu().long()

        for i in range(x_cpu.size(0)):
            lab = int(y_cpu[i].item())
            if 0 <= lab < K and len(xs_by[lab]) < n_per_label:
                xs_by[lab].append(x_cpu[i])
        if all(len(xs_by[k]) >= n_per_label for k in range(K)):
            break

    xs_plot, y_true_plot = [], []
    for k in range(K):
        xs = xs_by[k][:n_per_label]
        for x1 in xs:
            xs_plot.append(x1)
            y_true_plot.append(k)

    X = torch.stack(xs_plot, dim=0).to(device)
    out = model(X)
    x_hat = out["x_hat"].detach().cpu().numpy()
    x_np = X.detach().cpu().numpy()

    qy = out["q_y"].detach().cpu()
    y_pred = qy.argmax(dim=-1).numpy()
    conf = qy.max(dim=-1).values.numpy()

    def _name(k):
        if class_names and 0 <= k < len(class_names):
            return class_names[k]
        return str(k)

    rows, cols = K, n_per_label
    fig = plt.figure(figsize=(10, 2.6 * rows))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ax.plot(x_np[idx], label="x", linewidth=1.2)
            ax.plot(x_hat[idx], label="x_hat", linewidth=1.2)
            ax.grid(True, alpha=0.3)

            t = f"true {_name(y_true_plot[idx])} | pred {_name(int(y_pred[idx]))} ({conf[idx]:.2f})"
            ax.set_title(t, fontsize=9)
            if r == 0 and c == cols - 1:
                ax.legend(loc="upper right", fontsize=8)
            idx += 1
    return fig


@torch.no_grad()
def reco_error_profile_fig(model, loader, device, K: int, class_names=None):
    """
    Mean |x_hat-x| vs time per label, averaged per sample.
    """
    model.eval()
    x0, y0 = next(iter(loader))
    L = x0.shape[1]

    sum_err = torch.zeros(K, L)
    sum_n = torch.zeros(K)

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        err = (out["x_hat"] - x).abs().detach().cpu()  # [B,L]
        y = y.cpu().long()
        for k in range(K):
            m = (y == k)
            if m.any():
                sum_err[k] += err[m].sum(dim=0)
                sum_n[k] += float(m.sum().item())

    mean_err = sum_err / sum_n.clamp_min(1).unsqueeze(1)

    fig = plt.figure(figsize=(10, 3.5))
    for k in range(K):
        name = class_names[k] if class_names else f"class{k}"
        plt.plot(mean_err[k].numpy(), label=name)
    plt.title("Mean |reconstruction error| vs time")
    plt.xlabel("sample index")
    plt.ylabel("mean abs error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def component_usage_fig(comp: torch.Tensor, K: int, title="Component usage (argmax q_y)"):
    comp = comp.detach().cpu().numpy().astype(int)
    counts = np.bincount(comp, minlength=K)
    fig = plt.figure(figsize=(8, 3))
    plt.bar(np.arange(K), counts)
    plt.title(title)
    plt.xlabel("component")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)
    return fig, counts


def qy_stats(qy: torch.Tensor):
    qy = qy.detach().cpu()
    maxp = qy.max(dim=1).values
    eps = 1e-10
    ent = -(qy.clamp_min(eps) * torch.log(qy.clamp_min(eps))).sum(dim=1)
    return {
        "qy_max_mean": float(maxp.mean().item()),
        "qy_max_median": float(maxp.median().item()),
        "qy_entropy_mean": float(ent.mean().item()),
        "qy_entropy_median": float(ent.median().item()),
        "qy_mean_probs": to_numpy(qy.mean(dim=0)),
    }


def latent_pca_figs(mus: torch.Tensor, labels: torch.Tensor, K: int, title_prefix: str):
    Z2, W2, mean2 = pca_project(mus, 2)
    Z3, W3, mean3 = pca_project(mus, 3)

    z2 = Z2.numpy()
    z3 = Z3.numpy()
    lab = labels.detach().cpu().numpy().astype(int)

    fig2 = plt.figure(figsize=(6.5, 5.5))
    plt.scatter(z2[:, 0], z2[:, 1], c=lab, s=8, alpha=0.6)
    plt.title(f"{title_prefix} PCA-2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)

    fig3 = plt.figure(figsize=(7, 6))
    ax = fig3.add_subplot(111, projection="3d")
    ax.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=lab, s=8, alpha=0.6)
    ax.set_title(f"{title_prefix} PCA-3D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    return fig2, fig3


@torch.no_grad()
def latent_pca_with_prior_means_figs(model, mus, labels, device, K: int, title_prefix: str):
    model.eval()
    Z2, W2, mean2 = pca_project(mus, 2)
    Z3, W3, mean3 = pca_project(mus, 3)

    prior = model.prior()
    mu_k = prior["mu"].detach().cpu().float()  # [K,D]

    mu2 = (mu_k - mean2.unsqueeze(0)) @ W2
    mu3 = (mu_k - mean3.unsqueeze(0)) @ W3

    z2 = Z2.numpy()
    z3 = Z3.numpy()
    lab = labels.detach().cpu().numpy().astype(int)

    fig2 = plt.figure(figsize=(6.8, 5.8))
    plt.scatter(z2[:, 0], z2[:, 1], c=lab, s=8, alpha=0.55)
    plt.scatter(mu2[:, 0].numpy(), mu2[:, 1].numpy(), marker="X", s=220, edgecolors="k")
    for k in range(K):
        plt.text(float(mu2[k, 0]), float(mu2[k, 1]), f" μ{k}", fontsize=11, weight="bold")
    plt.title(f"{title_prefix} PCA-2D (with prior means)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)

    fig3 = plt.figure(figsize=(7.4, 6.4))
    ax = fig3.add_subplot(111, projection="3d")
    ax.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=lab, s=8, alpha=0.55)
    ax.scatter(mu3[:, 0].numpy(), mu3[:, 1].numpy(), mu3[:, 2].numpy(),
               marker="X", s=260, edgecolors="k")
    for k in range(K):
        ax.text(float(mu3[k, 0]), float(mu3[k, 1]), float(mu3[k, 2]), f" μ{k}", fontsize=11, weight="bold")
    ax.set_title(f"{title_prefix} PCA-3D (with prior means)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    return fig2, fig3


@torch.no_grad()
def sample_from_prior_fig(model, device, K: int, n_per_comp: int = 3):
    model.eval()
    prior = model.prior()
    mu = prior["mu"].to(device)
    logvar = prior["logvar"].to(device)
    D = mu.shape[1]

    zs = []
    for k in range(K):
        std = torch.exp(0.5 * logvar[k])
        eps = torch.randn(n_per_comp, D, device=device)
        z = mu[k].unsqueeze(0) + std.unsqueeze(0) * eps
        zs.append(z)

    Z = torch.cat(zs, dim=0)
    x_hat = model.decoder(Z).detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 2.2 * K))
    idx = 0
    for k in range(K):
        for j in range(n_per_comp):
            ax = fig.add_subplot(K, n_per_comp, k * n_per_comp + j + 1)
            ax.plot(x_hat[idx], linewidth=1.2)
            ax.set_title(f"y={k} sample {j}", fontsize=9)
            ax.grid(True, alpha=0.3)
            idx += 1
    return fig


@torch.no_grad()
def interpolate_fig(model, loader, device, class_id: int, steps: int = 8):
    model.eval()
    xs = []
    for x, y in loader:
        m = (y == class_id)
        if m.any():
            xs.append(x[m])
        if sum([t.shape[0] for t in xs]) >= 2:
            break
    if len(xs) == 0:
        return None
    xpair = torch.cat(xs, dim=0)[:2].to(device)

    enc = model.encoder(xpair)
    z0 = enc["mu_z"][0]
    z1 = enc["mu_z"][1]

    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    Z = (1 - alphas) * z0.unsqueeze(0) + alphas * z1.unsqueeze(0)
    x_hat = model.decoder(Z).detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 2.2))
    for i in range(steps):
        ax = fig.add_subplot(1, steps, i + 1)
        ax.plot(x_hat[i], linewidth=1.0)
        ax.set_title(f"{i}", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Latent interpolation (class {class_id})")
    return fig


# --------------------------
# Evaluation "dashboard"
# --------------------------
@torch.no_grad()
def log_full_dashboard(model, val_loader_metrics, val_loader_viz, device, K, class_names=None, step=0):
    model.eval()

    comp, qy, ytrue = gmvae.get_component_assignments(model, val_loader_metrics, device=device)

    fig_usage, _ = component_usage_fig(comp, K)
    stats_qy = qy_stats(qy)

    wandb.log({
        "dashboard/component_usage": fig_to_wandb_image(fig_usage, "Component usage"),
        "dashboard/qy_max_mean": stats_qy["qy_max_mean"],
        "dashboard/qy_max_median": stats_qy["qy_max_median"],
        "dashboard/qy_entropy_mean": stats_qy["qy_entropy_mean"],
        "dashboard/qy_entropy_median": stats_qy["qy_entropy_median"],
    }, step=step)

    fig = plt.figure(figsize=(8, 3))
    plt.bar(np.arange(K), stats_qy["qy_mean_probs"])
    plt.title("Mean q(y|x) probabilities over dataset")
    plt.xlabel("k")
    plt.ylabel("mean prob")
    plt.grid(True, axis="y", alpha=0.3)
    wandb.log({"dashboard/qy_mean_probs": fig_to_wandb_image(fig, "Mean q(y|x) probs")}, step=step)

    fig_reco = reco_grid_per_label(model, val_loader_metrics, device, K, n_per_label=3, class_names=class_names)
    wandb.log({"dashboard/reco_examples_per_label": fig_to_wandb_image(fig_reco, "reco per label")}, step=step)

    fig_err = reco_error_profile_fig(model, val_loader_metrics, device, K, class_names=class_names)
    wandb.log({"dashboard/reco_error_profile": fig_to_wandb_image(fig_err, "Mean abs error vs time")}, step=step)

    mus, comps_viz, ytrue_viz = collect_latent_mu(model, val_loader_viz, device=device, max_points=5000)

    fig2, fig3 = latent_pca_figs(mus, comps_viz, K, "Latent μ(x) colored by component")
    wandb.log({
        "dashboard/latent_pca2_comp": fig_to_wandb_image(fig2, "PCA2 component"),
        "dashboard/latent_pca3_comp": fig_to_wandb_image(fig3, "PCA3 component"),
    }, step=step)

    if ytrue_viz is not None:
        fig2, fig3 = latent_pca_figs(mus, ytrue_viz, K, "Latent μ(x) colored by true label")
        wandb.log({
            "dashboard/latent_pca2_true": fig_to_wandb_image(fig2, "PCA2 true"),
            "dashboard/latent_pca3_true": fig_to_wandb_image(fig3, "PCA3 true"),
        }, step=step)

        fig2, fig3 = latent_pca_with_prior_means_figs(model, mus, ytrue_viz, device, K, "Latent μ(x)")
        wandb.log({
            "dashboard/latent_pca2_true_with_means": fig_to_wandb_image(fig2, "PCA2 true + means"),
            "dashboard/latent_pca3_true_with_means": fig_to_wandb_image(fig3, "PCA3 true + means"),
        }, step=step)

    if ytrue is not None:
        wandb.log({
            "dashboard/confusion_true_vs_component": wandb.plot.confusion_matrix(
                probs=None,
                y_true=to_numpy(ytrue),
                preds=to_numpy(comp),
                class_names=[class_names[i] if class_names else f"class{i}" for i in range(K)],
                title="True label vs Component (argmax q_y)"
            )
        }, step=step)

    fig_samp = sample_from_prior_fig(model, device, K, n_per_comp=3)
    wandb.log({"dashboard/prior_samples": fig_to_wandb_image(fig_samp, "Samples from p(z|y) decoded")}, step=step)

    if ytrue is not None:
        for k in range(K):
            fig_int = interpolate_fig(model, val_loader_metrics, device, class_id=k, steps=8)
            if fig_int is not None:
                wandb.log({f"dashboard/interp_class_{k}": fig_to_wandb_image(fig_int, f"Interpolation class {k}")}, step=step)


# --------------------------
# Main training entrypoint
# --------------------------
def main(config: dict):

    DEBUG = True

    # NOTE: train_wandb() will call wandb.init() itself when use_wandb=True.
    # So we DO NOT init wandb here.
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load data ---
    def normalize_to_peak_amplitude(pulses):
        return pulses / np.max(pulses, axis=1, keepdims=True)

    def get_data(noise_level, Case='Case10', pathToDatasets='../synthetic_data/Synthetic_Datasets', validation=False):
        if validation:
            data = np.load(f"{pathToDatasets}/synthetic_validation_{Case}_480k_noise_{noise_level}.npz")
        else:
            data = np.load(f"{pathToDatasets}/synthetic_training_{Case}_120k_noise_{noise_level}.npz")
        return data["X"], data["y"]

    pathToDatasets = '../synthetic_data/Synthetic_Datasets_v03'
    Case = 'Case10'
    noise_level = 0.001

    X_train, Y_train = get_data(noise_level, Case=Case, pathToDatasets=pathToDatasets, validation=False)
    X_val, Y_val     = get_data(noise_level, Case=Case, pathToDatasets=pathToDatasets, validation=True)

    # IMPORTANT: train and val are not normalized -> the normalization is done here
    X_train_norm = normalize_to_peak_amplitude(X_train)
    X_val_norm   = normalize_to_peak_amplitude(X_val)

    class PulseDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
            self.X = torch.from_numpy(X).float()
            self.y = None if y is None else torch.from_numpy(y).long()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]

    train_ds = PulseDataset(X_train_norm, Y_train)
    val_ds   = PulseDataset(X_val_norm,   Y_val)

    train_loader       = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  drop_last=False)
    val_loader_metrics = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, drop_last=False)
    val_loader_viz     = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=True,  drop_last=False)

    # Quick check
    xb, yb = next(iter(train_loader))
    if DEBUG:
        print("batch x:", xb.shape, "batch y:", yb.shape, "unique y:", torch.unique(yb))

    L = X_train.shape[1]
    z_dim = (L // 8) if (config["z_dim"] is None) else int(config["z_dim"])

    model = gmvae.GMVAE(L=L, z_dim=z_dim, n_classes=config["K"], reco_loss=config["reco_loss"]).to(device)
    if DEBUG:
        print("L:", L, "z_dim:", z_dim, "K:", config["K"])
        print()

    # --- Let train_wandb handle training + wandb logging + dashboard calls ---
    gmvae.train_wandb(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader_metrics,          # METRICS loader
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=str(device),
        beta_z=config["beta_z"],
        beta_y=config["beta_y"],
        omega=config["omega"],
        label_mode=config["label_mode"],
        unlabeled_value=config["unlabeled_value"],
        print_every=config.get("print_every", 1),

        # W&B
        use_wandb=True,
        wandb_project=config["project"],
        wandb_run_name=config.get("run_name", None),
        wandb_config=config,                    # dump your config into W&B
        val_loader_viz=val_loader_viz,          # VIZ loader
        dashboard_every=config["dashboard_every"],
        class_names=config.get("class_names", None),
        K=config["K"],
    )

    # OPTIONAL: if you still want a local save as well (train_wandb already saves gmvae_model.pt)
    # (If you keep it, it will overwrite each run unless you change the filename.)
    # ckpt_path = "gmvae_model.pt"
    # torch.save({"model_state": model.state_dict()}, ckpt_path)


if __name__ == "__main__":
    config = dict(
        project="gmvae-psd",
        run_name=None,
        seed=0,
        K=3,
        batch_size=256,
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        beta_z=1.0,
        beta_y=1.0,
        omega=50.0,
        label_mode="full",
        unlabeled_value=-1,
        reco_loss="mse",
        z_dim=None,
        dashboard_every=2,
        class_names=["gamma", "neutron", "pileup"],
        print_every=1,
    )
    main(config)