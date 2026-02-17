# Minimal GMVAE (PyTorch): Encoder -> (q(y|x), q(z|x)), Decoder, Gaussian-mixture prior p(z|y), ELBO loss
# - No kNN
# - No triplet loss
# - Optional label loss can be added later
#
# Assumptions:
#   x is a 1D waveform of length L (float32)
#   If you use BCE, x must be in [0, 1]. Otherwise set reco_loss="mse".
#
# Usage:
#   model = GMVAE(L=296, z_dim=37, n_classes=3, reco_loss="bce")
#   train(model, train_loader, val_loader, epochs=50)

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from train_wandb import log_full_dashboard

# ---------------------------
# Utilities
# ---------------------------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) # external noise (if no noise z is deterministic)
    return mu + std * eps


def gaussian_kl_diag(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    """
    KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) )
    Returns: [batch]
    """
    # shape handling: all should broadcast to [B, D]
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 )
    kl = 0.5 * ( (logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0 )
    return kl.sum(dim=-1)  # [B]


def kl_categorical(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(q || p) for categorical distributions.
    q, p: [B, K], each row sums to 1
    Returns: [B]
    """
    # clam_min is to avoid issues with log(0)
    ## it ensures that q and p elements are always >= eps
    q = q.clamp_min(eps)  
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


# ---------------------------
# Model parts
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, L: int, z_dim: int, n_classes: int):
        super().__init__()
        h1 = L // 2
        h2 = L // 4

        self.fc1 = nn.Linear(L, h1)
        self.fc2 = nn.Linear(h1, h2)

        self.classifier = nn.Linear(h2, n_classes) 
        self.mu = nn.Linear(h2, z_dim) 
        self.logvar = nn.Linear(h2, z_dim) 

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, L]
        returns dict with:
          logits_y: [B, K]
          q_y: [B, K]   (y_prob)
          mu_z: [B, D]
          logvar_z: [B, D]
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        # discrete latent y
        ## decides the class
        logits_y = self.classifier(h) # = W.h + b
        q_y = F.softmax(logits_y, dim=-1)

        # continuous latent z
        ## Gaussain approximate posterior with mean mu and (log) varaiance logvar
        ## decides the position in the latent space given the chosen class
        mu_z = self.mu(h)
        logvar_z = self.logvar(h)

        return {"logits_y": logits_y, "q_y": q_y, "mu_z": mu_z, "logvar_z": logvar_z}

class Decoder(nn.Module):
    def __init__(self, L: int, z_dim: int, use_sigmoid: bool = True):
        super().__init__()
        h2 = L // 4
        h1 = L // 2

        self.fc1 = nn.Linear(z_dim, h2)
        self.fc2 = nn.Linear(h2, h1)
        self.fc3 = nn.Linear(h1, L)
        self.use_sigmoid = use_sigmoid

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D]
        returns x_hat: [B, L]
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_hat = self.fc3(h) # x_hat in ]-inf, +inf[]
        if self.use_sigmoid:
            x_hat = torch.sigmoid(x_hat) # x_hat in [0,1]
        return x_hat 


class GaussianMixturePrior(nn.Module):
    """
    p(y)=Cat(pi), p(z|y=k)=N(mu_k, diag(exp(logvar_k)))
    """
    def __init__(self, n_classes: int, z_dim: int, init_std: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.z_dim = z_dim

        # mixture weights pi_k
        self.pi_logits = nn.Parameter(torch.zeros(n_classes)) # learnable parameter

        # per-class Gaussian parameters
        self.mu = nn.Parameter(torch.zeros(n_classes, z_dim))
        # start with logvar = log(std^2)
        self.logvar = nn.Parameter(torch.full((n_classes, z_dim), 2.0 * torch.log(torch.tensor(init_std))))

    def p_y(self) -> torch.Tensor:
        """p(y): [K]"""
        return F.softmax(self.pi_logits, dim=-1)

    def forward(self) -> Dict[str, torch.Tensor]:
        """returns params (for convenience)"""
        return {"pi": self.p_y(), "mu": self.mu, "logvar": self.logvar}


# ---------------------------
# GMVAE wrapper + ELBO
# ---------------------------

@dataclass
class ELBOTerms:
    # per-sample terms (shape [B])
    reco: torch.Tensor
    kl_z: torch.Tensor
    kl_y: torch.Tensor


class GMVAE(nn.Module):
    def __init__(
        self,
        L: int,
        z_dim: int,
        n_classes: int,
        reco_loss: Literal["bce", "mse"] = "bce", 
        # BCE: Bernouilli likelihood -> requires inputs in [0, 1] (i.e. sigmoid)
        # MSE: Gaussian likelihood -> no need for bounded inputs
    ):
        super().__init__()
        use_sigmoid = (reco_loss == "bce")
        self.encoder = Encoder(L=L, z_dim=z_dim, n_classes=n_classes)
        self.decoder = Decoder(L=L, z_dim=z_dim, use_sigmoid=use_sigmoid)
        self.prior = GaussianMixturePrior(n_classes=n_classes, z_dim=z_dim)

        if reco_loss not in ("bce", "mse"):
            raise ValueError("reco_loss must be 'bce' or 'mse'")
        self.reco_loss = reco_loss
        self.L = L
        self.z_dim = z_dim
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        z = reparameterize(enc["mu_z"], enc["logvar_z"]) # differentiable function of mu and logvar
        x_hat = self.decoder(z)
        return {**enc, "z": z, "x_hat": x_hat}

    def elbo_terms(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> ELBOTerms:
        """
        ELBO = E_q [ -log p(x|z) ] + E_{q(y|x)} KL(q(z|x) || p(z|y)) + KL(q(y|x) || p(y))
        """
        x_hat = out["x_hat"]
        q_y = out["q_y"]            # [B, K]
        mu_z = out["mu_z"]          # [B, D]
        logvar_z = out["logvar_z"]  # [B, D]
        B = x.shape[0]

        # reconstruction term 
        if self.reco_loss == "bce":
            # BCE expects probabilities; x should be in [0,1] (x_hat already sigmoid)
            reco = F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=-1)  # [B]
        else:
            reco = F.mse_loss(x_hat, x, reduction="none").sum(dim=-1)  # [B]

        # KL_z: expected KL to mixture components under q(y|x)
        prior = self.prior()
        pi = prior["pi"]            # [K]
        mu_p = prior["mu"]          # [K, D]
        logvar_p = prior["logvar"]  # [K, D]

        # Compute KL for each component: [B, K]
        # Broadcast: mu_z/logvar_z -> [B, 1, D], mu_p/logvar_p -> [1, K, D]
        mu_z_b = mu_z.unsqueeze(1) # [B, 1, D]
        logvar_z_b = logvar_z.unsqueeze(1) # [B, 1, D]
        mu_p_b = mu_p.unsqueeze(0) # [1, K, D] 
        logvar_p_b = logvar_p.unsqueeze(0) # [1, K, D] 

        kl_each = gaussian_kl_diag(mu_z_b, logvar_z_b, mu_p_b, logvar_p_b)  # [B, K] via broadcast
        # But gaussian_kl_diag sums last dim; with broadcasting it returns [B, K]
        kl_z = (q_y * kl_each).sum(dim=-1)  # [B]

        # KL_y: encourage q(y|x) not to drift arbitrarily; match mixture weights p(y)
        p_y = pi.unsqueeze(0).expand(B, -1)  # [B, K]
        kl_y = kl_categorical(q_y, p_y)      # [B]

        # return terms of the loss function
        return ELBOTerms(reco=reco, kl_z=kl_z, kl_y=kl_y)

# ---------------------------
# Loss Funcion
# ---------------------------
@dataclass
class LossOutput:
    loss: torch.Tensor
    reco: torch.Tensor
    kl_z: torch.Tensor
    kl_y: torch.Tensor

def compute_loss(
    terms: ELBOTerms,
    beta_z: float = 1.0,
    beta_y: float = 1.0,
) -> LossOutput:
    """
    Assemble final scalar loss from per-sample ELBO terms.

    beta_z: Scales how strongly you force the latent Gaussian q(z|x) to look like the component prior p(z|y).
	    •	Larger beta_z → tighter, more “prior-shaped” latent space, sometimes better clustering but can hurt recostruction (posterior collapse if too strong).
	    •	Smaller beta_z → better reco, but latent space can get messy / less clustered.
    
    beta_y: Scales the categorical KL term
    	•	Larger beta_y → discourages component collapse and encourages using components in line with p(y) (often helps mixture usage).
        •	Too large can force overly uniform assignments even when data isn’t.
    
    TODO add:
      + omega * label_loss
      + gamma * triplet_loss
      + KL warmup schedules, etc.
    """
    elbo_per_sample = terms.reco + beta_z * terms.kl_z + beta_y * terms.kl_y
    loss = elbo_per_sample.mean()
    return LossOutput(
        loss=loss,
        reco=terms.reco.mean(),
        kl_z=terms.kl_z.mean(),
        kl_y=terms.kl_y.mean(),
    )

@dataclass
class LabelLossOutput:
    loss: torch.Tensor           # scalar
    n_used: int                  # how many samples contributed
    acc: Optional[float] = None  # accuracy on used samples (None if no samples)

def compute_label_loss(
    logits_y: torch.Tensor,
    y: Optional[torch.Tensor],
    mode: Literal["semi", "full", "off"] = "semi",
    unlabeled_value: int = -1,
    reduction: Literal["mean", "sum"] = "mean",
) -> LabelLossOutput:
    """
    Compute supervised cross-entropy on the classifier head.

    Args:
        logits_y: [B, K] raw class scores from encoder/classifier head.
        y:        [B] integer labels in [0, K-1], or unlabeled_value for unlabeled.
                  If y is None, behaves like mode="off".
        mode:
          - "semi": use only labeled samples (y != unlabeled_value). If none labeled, loss=0.
          - "full": require all samples labeled (no unlabeled_value). Uses all samples.
          - "off":  always returns loss=0.
        unlabeled_value: sentinel value marking unlabeled samples (only used in mode="semi").
        reduction (cross-entropy by default uses mean): 
          - "mean": average over used samples
          - "sum":  sum over used samples

    Returns:
        LabelLossOutput with scalar loss and n_used + optional accuracy.
    """
    device = logits_y.device
    if mode == "off" or y is None:
        return LabelLossOutput(loss=torch.zeros((), device=device), n_used=0, acc=None)

    y = y.to(device).long()

    if mode == "semi":
        mask = (y != unlabeled_value)
        n_used = int(mask.sum().item())
        if n_used == 0:
            return LabelLossOutput(loss=torch.zeros((), device=device), n_used=0, acc=None)

        logits_used = logits_y[mask]
        y_used = y[mask]

    elif mode == "full":
        if (y == unlabeled_value).any():
            # fail loudly: fully supervised means labels must exist for every sample
            raise ValueError(
                "compute_label_loss(mode='full') received unlabeled samples. "
                f"Found y == {unlabeled_value} in the batch."
            )
        logits_used = logits_y
        y_used = y
        n_used = int(y_used.numel())

    else:
        raise ValueError("mode must be one of: 'semi', 'full', 'off'")

    # CrossEntropyLoss expects logits (no softmax needed)
    # F.cross_entropy default reduction is 'mean'; we control it manually for accuracy reporting.
    per_sample = F.cross_entropy(logits_used, y_used, reduction="none")  # [n_used]

    if reduction == "mean":
        loss = per_sample.mean()
    elif reduction == "sum":
        loss = per_sample.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # Accuracy on the used samples
    preds = torch.argmax(logits_used, dim=-1)
    acc = float((preds == y_used).float().mean().item())

    return LabelLossOutput(loss=loss, n_used=n_used, acc=acc)

# ---------------------------
# Training loop (minimal)
# ---------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    beta_z: float = 1.0,
    beta_y: float = 1.0,
    omega: float = 0.0,
    label_mode: Literal["semi", "full", "off"] = "off",
    unlabeled_value: int = -1,
) -> Dict[str, float]:
    """
    Train one epoch.

    Supports:
      - Pure ELBO training: omega=0.0 OR label_mode="off"
      - Semi-supervised:   omega>0 and label_mode="semi" with unlabeled_value sentinel
      - Fully supervised:  omega>0 and label_mode="full" (requires all y labeled)

    Expected batches:
      - (x,) or x
      - (x, y)  where y is [B] long tensor

    Returns epoch averages for:
      loss_total, loss_elbo, reco, kl_z, kl_y, label_loss, label_acc, n_labeled_used
    """
    model.train()
    totals = {
        "loss_total": 0.0,
        "loss_elbo": 0.0,
        "reco": 0.0,
        "kl_z": 0.0,
        "kl_y": 0.0,
        "label_loss": 0.0,
        "label_acc_sum": 0.0,   # weighted by n_used
        "n_labeled_used": 0.0,  # total labeled samples used across epoch
    }
    n_samples = 0

    totals_pred = {
        "pred_acc_sum": 0.0,     # weighted by n_used
        "pred_conf_sum": 0.0,    # weighted by n_used
        "pred_ent_sum": 0.0,     # weighted by n_used
        "n_pred_used": 0.0,
    }

    for batch in loader:
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
        else:
            x, y = batch, None

        x = x.to(device)
        if y is not None:
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward + ELBO terms
        out = model(x)
        terms = model.elbo_terms(x, out)
        elbo_out = compute_loss(terms, beta_z=beta_z, beta_y=beta_y)

        # --- prediction diagnostics (DOES NOT affect training) ---
        pstats = _prediction_stats_from_logits(
            logits_y=out["logits_y"].detach(),
            y=y.detach() if y is not None else None,
            mode=label_mode,                     # uses your current mode for masking unlabeled
            unlabeled_value=unlabeled_value,
        )
        if pstats["n_used"] > 0:
            n_used = pstats["n_used"]
            totals_pred["pred_acc_sum"]  += pstats["acc"] * n_used
            totals_pred["pred_conf_sum"] += pstats["conf_mean"] * n_used
            totals_pred["pred_ent_sum"]  += pstats["entropy_mean"] * n_used
            totals_pred["n_pred_used"]   += n_used

        # Optional label loss
        use_label = (omega != 0.0) and (label_mode != "off")
        if use_label:
            lab = compute_label_loss(
                logits_y=out["logits_y"],
                y=y,
                mode=label_mode,
                unlabeled_value=unlabeled_value,
                reduction="mean",
            )
            total_loss = elbo_out.loss + omega * lab.loss
        else:
            lab = None
            total_loss = elbo_out.loss

        total_loss.backward()
        optimizer.step()

        bs = x.size(0)
        n_samples += bs

        # Accumulate (sample-weighted for ELBO metrics)
        totals["loss_total"] += float(total_loss.item()) * bs
        totals["loss_elbo"] += float(elbo_out.loss.item()) * bs
        totals["reco"] += float(elbo_out.reco.item()) * bs
        totals["kl_z"] += float(elbo_out.kl_z.item()) * bs
        totals["kl_y"] += float(elbo_out.kl_y.item()) * bs

        # Accumulate label stats (weighted by number of labeled used, not batch size)
        if lab is not None and lab.n_used > 0:
            totals["label_loss"] += float(lab.loss.item()) * lab.n_used
            totals["label_acc_sum"] += float(lab.acc) * lab.n_used if lab.acc is not None else 0.0
            totals["n_labeled_used"] += float(lab.n_used)

    # Finalize averages
    out_stats = {
        "loss_total": totals["loss_total"] / max(n_samples, 1),
        "loss_elbo": totals["loss_elbo"] / max(n_samples, 1),
        "reco": totals["reco"] / max(n_samples, 1),
        "kl_z": totals["kl_z"] / max(n_samples, 1),
        "kl_y": totals["kl_y"] / max(n_samples, 1),
    }

    # Label averages (over labeled used)
    n_lab = totals["n_labeled_used"]
    if n_lab > 0:
        out_stats["label_loss"] = totals["label_loss"] / n_lab
        out_stats["label_acc"] = totals["label_acc_sum"] / n_lab
        out_stats["n_labeled_used"] = n_lab
    else:
        out_stats["label_loss"] = 0.0
        out_stats["label_acc"] = float("nan")
        out_stats["n_labeled_used"] = 0.0

    # --- add prediction diagnostics to output ---
    npu = totals_pred["n_pred_used"]
    if npu > 0:
        out_stats["pred_acc"] = totals_pred["pred_acc_sum"] / npu
        out_stats["pred_conf"] = totals_pred["pred_conf_sum"] / npu
        out_stats["pred_entropy"] = totals_pred["pred_ent_sum"] / npu
        out_stats["n_pred_used"] = npu
    else:
        out_stats["pred_acc"] = float("nan")
        out_stats["pred_conf"] = float("nan")
        out_stats["pred_entropy"] = float("nan")
        out_stats["n_pred_used"] = 0.0

    return out_stats

# ---------------------------
# Evaluation
# ---------------------------

@torch.no_grad()
def evaluate_losses(
    model,
    loader,
    device: torch.device,
    beta_z: float = 1.0,
    beta_y: float = 1.0,
    omega: float = 0.0,
    label_mode: Literal["semi", "full", "off"] = "off",
    unlabeled_value: int = -1,
) -> Dict[str, float]:
    """
    Validation under the TRAINING OBJECTIVE:
      loss_total = ELBO + omega * label_loss (optional)
      ELBO = reco + beta_z * kl_z + beta_y * kl_y

    Returns averages for:
      loss_total, loss_elbo, reco, kl_z, kl_y, label_loss, label_acc, n_labeled_used
    PLUS (if y available):
      pred_acc, pred_conf, pred_entropy, n_pred_used
    """
    model.eval()

    totals = dict(
        loss_total=0.0,
        loss_elbo=0.0,
        reco=0.0,
        kl_z=0.0,
        kl_y=0.0,
        label_loss=0.0,        # summed over labeled examples
        label_acc_sum=0.0,     # summed over labeled examples
        n_labeled_used=0.0,
    )
    n_samples = 0

    totals_pred = dict(
        pred_acc_sum=0.0,      # weighted by n_used
        pred_conf_sum=0.0,     # weighted by n_used
        pred_ent_sum=0.0,      # weighted by n_used
        n_pred_used=0.0,
    )

    use_label = (omega != 0.0) and (label_mode != "off")

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
        else:
            x, y = batch, None

        x = x.to(device)
        if y is not None:
            y = y.to(device)

        out = model(x)
        terms = model.elbo_terms(x, out)
        elbo_out = compute_loss(terms, beta_z=beta_z, beta_y=beta_y)  # per-batch mean scalars

        # --- prediction diagnostics (no effect on loss) ---
        # In semi mode: ignore unlabeled_value
        # Otherwise: use labels if provided (still ignores unlabeled_value if present)
        diag_mode = "semi" if label_mode == "semi" else "off"
        pstats = _prediction_stats_from_logits(
            logits_y=out["logits_y"],
            y=y,
            mode=diag_mode,
            unlabeled_value=unlabeled_value,
        )
        if pstats["n_used"] > 0:
            n_used = pstats["n_used"]
            totals_pred["pred_acc_sum"]  += pstats["acc"] * n_used
            totals_pred["pred_conf_sum"] += pstats["conf_mean"] * n_used
            totals_pred["pred_ent_sum"]  += pstats["entropy_mean"] * n_used
            totals_pred["n_pred_used"]   += n_used

        # total loss scalar (batch mean)
        total_loss = elbo_out.loss
        lab = None
        if use_label:
            lab = compute_label_loss(
                logits_y=out["logits_y"],
                y=y,
                mode=label_mode,
                unlabeled_value=unlabeled_value,
                reduction="mean",  # mean over used labels (or 0 if none used)
            )
            total_loss = total_loss + omega * lab.loss

        bs = x.size(0)
        n_samples += bs

        # multiply by bs to accumulate per-sample averages across batches
        totals["loss_total"] += float(total_loss.item()) * bs
        totals["loss_elbo"] += float(elbo_out.loss.item()) * bs
        totals["reco"] += float(elbo_out.reco.item()) * bs
        totals["kl_z"] += float(elbo_out.kl_z.item()) * bs
        totals["kl_y"] += float(elbo_out.kl_y.item()) * bs

        if lab is not None and lab.n_used > 0:
            # lab.loss is mean over n_used, so sum = mean * n_used
            totals["label_loss"] += float(lab.loss.item()) * float(lab.n_used)
            if lab.acc is not None:
                totals["label_acc_sum"] += float(lab.acc) * float(lab.n_used)
            totals["n_labeled_used"] += float(lab.n_used)

    out_stats = {
        "loss_total": totals["loss_total"] / max(n_samples, 1),
        "loss_elbo": totals["loss_elbo"] / max(n_samples, 1),
        "reco": totals["reco"] / max(n_samples, 1),
        "kl_z": totals["kl_z"] / max(n_samples, 1),
        "kl_y": totals["kl_y"] / max(n_samples, 1),
    }

    n_lab = totals["n_labeled_used"]
    if n_lab > 0:
        out_stats["label_loss"] = totals["label_loss"] / n_lab
        out_stats["label_acc"] = totals["label_acc_sum"] / n_lab
        out_stats["n_labeled_used"] = n_lab
    else:
        out_stats["label_loss"] = 0.0
        out_stats["label_acc"] = float("nan")
        out_stats["n_labeled_used"] = 0.0

    # --- finalize prediction diagnostics ---
    npu = totals_pred["n_pred_used"]
    if npu > 0:
        out_stats["pred_acc"] = totals_pred["pred_acc_sum"] / npu
        out_stats["pred_conf"] = totals_pred["pred_conf_sum"] / npu
        out_stats["pred_entropy"] = totals_pred["pred_ent_sum"] / npu
        out_stats["n_pred_used"] = npu
    else:
        out_stats["pred_acc"] = float("nan")
        out_stats["pred_conf"] = float("nan")
        out_stats["pred_entropy"] = float("nan")
        out_stats["n_pred_used"] = 0.0

    return out_stats

@torch.no_grad()
def evaluate_predictions(model, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    preds, confs, trues = [], [], []
    ent_sum = 0.0
    n_ent = 0

    eps = 1e-10

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
        else:
            x, y = batch, None

        x = x.to(device)
        enc = model.encoder(x)
        qy = enc["q_y"].detach().cpu()                 # [B,K]

        pred = torch.argmax(qy, dim=-1)
        conf = torch.max(qy, dim=-1).values
        ent = -(qy.clamp_min(eps) * torch.log(qy.clamp_min(eps))).sum(dim=-1)  # [B]

        preds.append(pred)
        confs.append(conf)
        ent_sum += float(ent.sum().item())
        n_ent += int(ent.numel())

        if y is not None:
            trues.append(y.detach().cpu().long())

    y_pred = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    conf = torch.cat(confs, dim=0) if confs else torch.empty(0, dtype=torch.float32)
    y_true = torch.cat(trues, dim=0) if trues else None

    pred_conf = float(conf.mean().item()) if conf.numel() > 0 else float("nan")
    pred_entropy = float(ent_sum / max(n_ent, 1))

    if y_true is not None and y_true.numel() > 0:
        pred_acc = float((y_pred == y_true).float().mean().item())
        n_used = int(y_true.numel())
    else:
        pred_acc = float("nan")
        n_used = 0

    return {
        "y_pred": y_pred,
        "conf": conf,
        "y_true": y_true,
        "pred_acc": pred_acc,
        "pred_conf": pred_conf,
        "pred_entropy": pred_entropy,
        "n_pred_used": n_used,
    }


# ---------------------------
# Evaluation wrapper
# ---------------------------
@torch.no_grad()
def evaluate_all(
    model,
    loader_metrics,
    loader_preds=None,
    device: torch.device = None,
    beta_z: float = 1.0,
    beta_y: float = 1.0,
    omega: float = 0.0,
    label_mode: Literal["semi", "full", "off"] = "off",
    unlabeled_value: int = -1,
) -> Dict[str, object]:
    """
    Runs:
      - evaluate_losses on loader_metrics
      - evaluate_predictions on loader_preds (defaults to loader_metrics)
    """
    if device is None:
        device = next(model.parameters()).device
    if loader_preds is None:
        loader_preds = loader_metrics

    losses = evaluate_losses(
        model=model,
        loader=loader_metrics,
        device=device,
        beta_z=beta_z,
        beta_y=beta_y,
        omega=omega,
        label_mode=label_mode,
        unlabeled_value=unlabeled_value,
    )
    preds = evaluate_predictions(
        model=model,
        loader=loader_preds,
        device=device,
    )
    return {**losses, **preds}

# ---------------------------
# Training loop (with wandb)
# ---------------------------

def train_wandb(
    model,
    train_loader,
    val_loader=None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    beta_z: float = 1.0,
    beta_y: float = 1.0,
    omega: float = 0.0,
    label_mode: Literal["semi", "full", "off"] = "off",
    unlabeled_value: int = -1,
    print_every: int = 1,

    # ---------- W&B ----------
    use_wandb: bool = False,
    wandb_project: str = "gmvae-psd",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    val_loader_viz=None,                 # shuffled loader for PCA plots
    dashboard_every: int = 5,            # heavy plots every N epochs
    class_names: Optional[list[str]] = None,
    K: Optional[int] = None,             # number of classes/components, for plots/confusions
) -> "GMVAE":
    """
    optional Weights & Biases logging.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    model.to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # W&B init (optional)
    if use_wandb:
        import wandb
        cfg = dict(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            beta_z=beta_z,
            beta_y=beta_y,
            omega=omega,
            label_mode=label_mode,
            unlabeled_value=unlabeled_value,
            device=str(device_t),
        )
        if wandb_config:
            cfg.update(wandb_config)

        wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)

        # nice to track model gradients/parameters
        # (optional, comment out if you find it heavy)
        wandb.watch(model, log="gradients", log_freq=max(10, dashboard_every))

    # Header printer (unchanged)
    def _hdr():
        if val_loader is not None:
            return (
                "epoch | "
                "train_total train_elbo  reco   klz    kly   lab_loss lab_acc n_lab || "
                "val_total   val_elbo    reco   klz    kly   lab_loss lab_acc n_lab"
            )
        else:
            return "epoch | train_total train_elbo reco klz kly lab_loss lab_acc n_lab"
    print(_hdr())

    # infer K if not given
    if K is None:
        K = getattr(model, "n_classes", None)

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_t,
            beta_z=beta_z,
            beta_y=beta_y,
            omega=omega,
            label_mode=label_mode,
            unlabeled_value=unlabeled_value,
        )

        va = None
        if val_loader is not None:
            # evaluate_all returns loss stats + y_pred/conf/y_true/acc
            va = evaluate_all(
                model=model,
                loader_metrics=val_loader,         # shuffle=False recommended
                loader_preds=val_loader,           # use same loader for pred metrics/confusions
                device=device_t,
                beta_z=beta_z,
                beta_y=beta_y,
                omega=omega,
                label_mode=label_mode,
                unlabeled_value=unlabeled_value,
            )

        # ------------------------------
        # Console printing (as before)
        # ------------------------------
        if (epoch % print_every) == 0:
            if va is not None:
                print(
                    f"{epoch:5d} | "
                    f"{tr['loss_total']:10.4f} {tr['loss_elbo']:10.4f} {tr['reco']:6.3f} "
                    f"{tr['kl_z']:6.3f} {tr['kl_y']:6.3f} "
                    f"{tr['label_loss']:8.4f} {tr['label_acc'] if tr['label_acc']==tr['label_acc'] else float('nan'):7.3f} {int(tr['n_labeled_used']):5d} || "
                    f"{va['loss_total']:10.4f} {va['loss_elbo']:10.4f} {va['reco']:6.3f} "
                    f"{va['kl_z']:6.3f} {va['kl_y']:6.3f} "
                    f"{va['label_loss']:8.4f} {va['label_acc'] if va['label_acc']==va['label_acc'] else float('nan'):7.3f} {int(va['n_labeled_used']):5d}"
                )
            else:
                print(
                    f"{epoch:5d} | "
                    f"{tr['loss_total']:10.4f} {tr['loss_elbo']:10.4f} {tr['reco']:6.3f} "
                    f"{tr['kl_z']:6.3f} {tr['kl_y']:6.3f} "
                    f"{tr['label_loss']:8.4f} {tr['label_acc'] if tr['label_acc']==tr['label_acc'] else float('nan'):7.3f} {int(tr['n_labeled_used']):5d}"
                )

        # ------------------------------
        # W&B scalar logging (every epoch)
        # ------------------------------
        if use_wandb:
            import wandb

            log_dict = {
                "epoch": epoch,

                "train/loss_total": tr["loss_total"],
                "train/loss_elbo": tr["loss_elbo"],
                "train/reco": tr["reco"],
                "train/kl_z": tr["kl_z"],
                "train/kl_y": tr["kl_y"],
                "train/label_loss": tr.get("label_loss", 0.0),
                "train/label_acc": tr.get("label_acc", np.nan),
                "train/n_labeled_used": tr.get("n_labeled_used", 0),

                "train/pred_acc": tr.get("pred_acc", np.nan),
                "train/pred_conf": tr.get("pred_conf", np.nan),
                "train/pred_entropy": tr.get("pred_entropy", np.nan),
                "train/n_pred_used": tr.get("n_pred_used", 0)
                
            }

            if va is not None:
                log_dict.update({
                    "val/loss_total": va["loss_total"],
                    "val/loss_elbo": va["loss_elbo"],
                    "val/reco": va["reco"],
                    "val/kl_z": va["kl_z"],
                    "val/kl_y": va["kl_y"],
                    "val/label_loss": va.get("label_loss", 0.0),
                    "val/label_acc": va.get("label_acc", np.nan),
                    "val/n_labeled_used": va.get("n_labeled_used", 0),

                    "val/pred_acc": va.get("pred_acc", np.nan),
                    "val/pred_conf": va.get("pred_conf", np.nan),
                    "val/pred_entropy": va.get("pred_entropy", np.nan)
                })

            wandb.log(log_dict, step=epoch)

            # ------------------------------
            # W&B dashboard logging (heavy plots)
            # ------------------------------
            do_dash = (dashboard_every is not None) and (
                epoch == 1 or epoch == epochs or (epoch % dashboard_every) == 0
            )
            if do_dash and (val_loader is not None):
                # Use a shuffled loader for PCA if provided; else fall back to val_loader
                vl_viz = val_loader_viz if val_loader_viz is not None else val_loader

                # This function should create & wandb.log() images/plots.
                # Provide K and class_names so it can label axes.
                log_full_dashboard(
                    model=model,
                    val_loader_metrics=val_loader,     # shuffle=False ok
                    val_loader_viz=vl_viz,             # shuffle=True recommended
                    device=device_t,
                    K=K,
                    class_names=class_names,
                    step=epoch,
                )

    if use_wandb:
        import wandb
        # Optionally save checkpoint as an artifact
        ckpt_path = "gmvae_model.pt"
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        wandb.save(ckpt_path)

    return model


# ---------------------------
# Quick inference helpers
# ---------------------------

@torch.no_grad()
def _prediction_stats_from_logits(
    logits_y: torch.Tensor,                          # [B,K]
    y: Optional[torch.Tensor],                       # [B] or None
    mode: Literal["semi", "full", "off"] = "off",
    unlabeled_value: int = -1,
) -> Dict[str, object]:
    """
    Compute prediction stats from logits (or q_y derived from them).
    Returns dict with:
      - n_used (int)
      - acc (float or None)
      - conf_mean (float or None)
      - entropy_mean (float or None)
    Notes:
      - If y is None -> n_used=0 and stats None
      - If mode="semi" -> exclude y==unlabeled_value
      - If mode="full" -> require all labeled; if unlabeled present, they are excluded anyway
      - If mode="off" -> still compute over all labeled y provided (best for diagnostics)
    """
    if y is None:
        return {"n_used": 0, "acc": None, "conf_mean": None, "entropy_mean": None}

    y = y.long()
    if mode == "semi":
        mask = (y != unlabeled_value)
    else:
        # "full" or "off": if y includes unlabeled_value, exclude it anyway
        mask = (y != unlabeled_value)

    if mask.sum().item() == 0:
        return {"n_used": 0, "acc": None, "conf_mean": None, "entropy_mean": None}

    logits = logits_y[mask]                          # [n_used,K]
    y_used = y[mask]                                 # [n_used]

    qy = torch.softmax(logits, dim=-1)               # [n_used,K]
    pred = torch.argmax(qy, dim=-1)                  # [n_used]
    conf = torch.max(qy, dim=-1).values              # [n_used]

    eps = 1e-10
    ent = -(qy.clamp_min(eps) * torch.log(qy.clamp_min(eps))).sum(dim=-1)  # [n_used]

    acc = (pred == y_used).float().mean().item()

    return {
        "n_used": int(mask.sum().item()),
        "acc": float(acc),
        "conf_mean": float(conf.mean().item()),
        "entropy_mean": float(ent.mean().item()),
    }

@torch.no_grad()
def predict_membership(model: GMVAE, x: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """Returns q(y|x) as [B, K]"""
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    out = model.encoder(x)
    return out["q_y"]


@torch.no_grad()
def encode_mu(model: GMVAE, x: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """Returns latent mean mu(x) as [B, D]"""
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    out = model.encoder(x)
    return out["mu_z"]


# ---------------------------
# Inspection
# ---------------------------

@torch.no_grad()
def inspect_prior(model):
    model.eval()
    prior = model.prior()
    pi = prior["pi"].detach().cpu()                 # [K]
    mu = prior["mu"].detach().cpu()                 # [K, D]
    logvar = prior["logvar"].detach().cpu()         # [K, D]
    std = torch.exp(0.5 * logvar)                   # [K, D]

    print("Mixture weights p(y)=pi:")
    for k, pik in enumerate(pi.tolist()):
        print(f"  k={k}: pi={pik:.4f}")

    print("\nComponent means mu_k: (showing first 8 dims)")
    for k in range(mu.shape[0]):
        vals = mu[k, :8].tolist()
        print(f"  k={k}: {['%+.3f'%v for v in vals]}")

    print("\nComponent std sigma_k: (showing first 8 dims)")
    for k in range(std.shape[0]):
        vals = std[k, :8].tolist()
        print(f"  k={k}: {['%.3f'%v for v in vals]}")

@torch.no_grad()
def get_component_assignments(model, loader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_comp = []
    all_qy = []
    all_ytrue = []

    for batch in loader:
        # expects batch = (x, y) or (x,)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            all_ytrue.append(y.cpu())
        else:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch

        x = x.to(device)
        enc = model.encoder(x)
        qy = enc["q_y"].detach().cpu()        # [B, K]
        comp = torch.argmax(qy, dim=-1)       # [B]

        all_qy.append(qy)
        all_comp.append(comp)

    all_comp = torch.cat(all_comp, dim=0)
    all_qy = torch.cat(all_qy, dim=0)
    all_ytrue = torch.cat(all_ytrue, dim=0) if len(all_ytrue) else None
    return all_comp, all_qy, all_ytrue

def confusion_true_vs_component(y_true, comp, n_classes=None, n_components=None):
    y_true = y_true.long()
    comp = comp.long()

    if n_classes is None:
        n_classes = int(y_true.max().item()) + 1
    if n_components is None:
        n_components = int(comp.max().item()) + 1

    cm = torch.zeros(n_classes, n_components, dtype=torch.int64)
    for yt, ck in zip(y_true.tolist(), comp.tolist()):
        cm[yt, ck] += 1
    return cm

def print_confusion(cm, class_names=None, comp_names=None):
    n_classes, n_comp = cm.shape
    if class_names is None:
        class_names = [f"class{c}" for c in range(n_classes)]
    if comp_names is None:
        comp_names = [f"k{c}" for c in range(n_comp)]

    print("Confusion: rows=true class, cols=component k")
    header = " " * 12 + " ".join([f"{name:>8}" for name in comp_names])
    print(header)
    for i in range(n_classes):
        row = " ".join([f"{cm[i,j].item():8d}" for j in range(n_comp)])
        print(f"{class_names[i]:>12} {row}")

