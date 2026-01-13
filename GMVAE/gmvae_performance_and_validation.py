import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
from tabulate import tabulate as tab
# import types # to use dictionary as an object

# !pip3 install torch torchvision 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# !pip3 install umap-learn
# import umap

#!pip3 install plotly
import plotly.graph_objects as go # interactive plots

import wandb
import io
from PIL import Image


class GMVAEAnalyzer:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def _collect(
        self,
        return_y      = False,
        return_x      = False, 
        return_logits = False,
        return_probs  = False,
        return_preds  = False,
        return_latent = False,
        return_reco   = False,
    ):
        self.model.eval()

        out = {"y_true": []}
        if return_x:      out["x"]      = []
        if return_logits: out["logits"] = []
        if return_probs:  out["probs"]  = []
        if return_preds:  out["preds"]  = []
        if return_latent: out["z"]      = []
        if return_reco:   out["x_hat"]  = []

        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)

                x_hat, logits, mu, logvar = self.model(x)

                if return_y:
                    out["y_true"].append(y.cpu().numpy())

                if return_x:
                    out["x"].append(x.cpu().numpy())

                if return_logits:
                    out["logits"].append(logits.cpu().numpy())
                # Softmax(logits) -> cls. output in [0, 1]
                if return_probs: 
                    probs = torch.softmax(logits, dim=1)
                    out["probs"].append(probs.cpu().numpy())
                # Argmax(logits) -> predicted classes   
                # - logits gives 3 scores (one of each class)
                # - here we say that the prediction corresponds to the higher class
                if return_preds:
                    preds = torch.argmax(logits, dim=1)
                    out["preds"].append(preds.cpu().numpy())

                if return_latent:
                    out["z"].append(mu.cpu().numpy())

                if return_reco:
                    out["x_hat"].append(x_hat.cpu().numpy())

        for k in out:
            if k in ["y_true", "preds"]:
                out[k] = np.hstack(out[k])
            else:
                out[k] = np.vstack(out[k])

        return out
    
#---------------------------------
#       Helper Functions
#---------------------------------

def get_categories_dict():
    return {
            'n': 1, # neutron
            'g': 0, # gamma
            'p': 2 # pile-up
              } 

def get_names():
    names  = {'g': 'γ', 'n': 'n', 'p': 'Pile-up'}
    return names

#---------------------------------
#       Loss Function
#---------------------------------

def plot_loss_function(loss, title='total loss = MSE + KL + (alpha * CE)', logScale=True, fig_size=(8, 5)):
    _, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_title(title)
    ax.plot(loss)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    if logScale:
        ax.set_yscale('log')
    ax.grid()
    plt.tight_layout()

def plot_loss_functions(total_loss, total_reco, total_kl, total_ce, fig_size=(10, 5)):
    _, axs = plt.subplots(2, 2, figsize=fig_size)
    axs = axs.flatten()
    axs[0].set_title(f'total loss = MSE + KL + (alpha * CE)')
    axs[1].set_title(f'Reconstruction loss (MSE)')
    axs[2].set_title(f'KL divergence loss (KL)')
    axs[3].set_title(f'Cross entropy loss CE')

    axs[0].plot(total_loss)
    axs[1].plot(total_reco)
    axs[2].plot(total_kl)
    axs[3].plot(total_ce)


    for i in range(0, 4):
        axs[i].set_xlabel("Epoch", fontsize=12)
        axs[i].set_ylabel("Loss", fontsize=12)
        if i!=1:
            axs[i].set_yscale('log')
        axs[i].grid()
    plt.tight_layout()

#---------------------------------
#      Confusion Matrix
#---------------------------------

def plot_confusion_matrix_from_arrays(y_true, y_pred, normalize=True, title="Normalized Confusion Matrix"):
    """
    y_true : np.ndarray (N,)
    y_pred : np.ndarray (N,)
    """
    cat_dict = get_categories_dict()  # {'g': 0, 'n': 1, 'p': 2}
    name_map = get_names()            # {'g': 'γ', 'n': 'n', 'p': 'Pile-up'}
    ## need to sort everything according to y_pred 
    sorted_keys = sorted(cat_dict, key=cat_dict.get)
    labels_order = [cat_dict[k] for k in sorted_keys] # not really necessary but apparently it avoids bugs...
    display_labels = [name_map[k] for k in sorted_keys]

    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, labels=labels_order, normalize=norm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".1%" if normalize else "d", colorbar=True)

    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    return fig

def plot_confusion_matrix(analyzer):
    """
    using GMVAEAnalyzer
    """

    out = analyzer._collect(
        return_y=True,
        return_preds=True
    )

    y_true = out["y_true"]
    y_pred = out["preds"]

    plot_confusion_matrix_from_arrays(y_true, y_pred)

#---------------------------------
#      Reconstruction
#---------------------------------

def reconstruction_error_from_arrays(x_hat, x_sample, y_sample):
    """
    Get total reconstruction error and errors for each categories
    """
    # Calculate overall error (aka reconstruction error)
    overall_mse = np.mean((x_sample - x_hat)**2).item()
    print(f"Overall MSE: {overall_mse:.6f}")

    # Calculate error PER CATEGORY 
    for label, name in enumerate(['Neutron', 'Gamma', 'Pile-up']):
        mask = (y_sample == label)
        if mask.sum() > 0:
            cat_mse = np.mean((x_sample[mask] - x_hat[mask])**2).item()
            print(f" {name} Reconstruction Error: {cat_mse:.6f}")

def reconstruction_error(analyzer):
    """
    using GMVAEAnalyzer
    """
    out = analyzer._collect(
        return_y=True,
        return_x=True,
        return_reco=True
    )
    x_hat = out["x_hat"]
    x_sample = out["x"]
    y_sample = out["y_true"]
    reconstruction_error_from_arrays(x_hat, x_sample, y_sample)

def plot_random_reconstructions_from_arrays(x, x_hat, y_true, y_pred, num_samples=6, class_names=('γ', 'n', 'p'), seed=None):
    """
    x, x_hat : np.ndarray (N, T)
    y_true   : np.ndarray (N,)
    y_pred   : np.ndarray (N,)
    """

    if seed is not None:
        np.random.seed(seed)

    idxs = np.random.choice(len(x), num_samples, replace=False)

    nrows = int(np.ceil(num_samples / 3))
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        ax.plot(x[idx], label="Original", linewidth=1.5)
        ax.plot(x_hat[idx], "--", label="Reconstructed", linewidth=1.5)

        correct = (y_true[idx] == y_pred[idx])
        title_color = "darkgreen" if correct else "darkred"

        ax.set_title(
            f"True: {class_names[y_true[idx]]} | "
            f"Pred: {class_names[y_pred[idx]]}",
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )

        ax.set_xlabel("Time (Samples)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused axes
    for ax in axes[len(idxs):]:
        ax.axis("off")

    axes[0].legend(loc="upper right")
    plt.tight_layout()

    return fig

def plot_random_reconstructions(analyzer, num_samples=6, seed=None):
    out = analyzer._collect(
        return_x=True,
        return_y=True,
        return_reco=True,
        return_preds=True,
    )

    plot_random_reconstructions_from_arrays(
        x=out["x"],
        x_hat=out["x_hat"],
        y_true=out["y_true"],
        y_pred=out["preds"],
        num_samples=num_samples,
        seed=seed,
    )

#---------------------------------
#      Latent Space
#---------------------------------
def plot_2d_pca_from_arrays(z, y, class_names=('γ', 'n', 'p'),):
    """
    z : np.ndarray (N, latent_dim)
    y : np.ndarray (N,)
    """

    # --- 2D PCA ---
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    evar = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#007bff', '#ff7f0e', '#a9a9a9']
    opacities = [0.7, 0.7, 0.2]

    for i, name in enumerate(class_names):
        mask = (y == i)
        ax.scatter(
            z_pca[mask, 0],
            z_pca[mask, 1],
            s=2,
            alpha=opacities[i],
            color=colors[i],
            label=name,
        )

    # --- Explained variance box ---
    stats_text = (
        f"Explained Variance\n"
        f"PC1: {evar[0]:.1%}\n"
        f"PC2: {evar[1]:.1%}\n"
        f"────────────────\n"
        f"Total: {evar.sum():.1%}"
    )

    ax.text(
        1.05, 0.65,
        stats_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.8,
            edgecolor='gray'
        )
    )

    # --- Formatting ---
    ax.set_title("2D PCA Projection (Latent Space)", fontsize=14, pad=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(
        title="Particle Type",
        markerscale=5,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
    )

    plt.tight_layout()
    return fig

def plot_2d_pca(analyzer):
    """
    PCA of GMVAE latent space using analyzer
    """

    out = analyzer._collect(
        return_y=True,
        return_latent=True,
    )

    plot_2d_pca_from_arrays(
        z=out["z"],
        y=out["y_true"],
    )

def plot_3d_pca_projections_from_arrays(z, y, draw_priority = ['p', 'g', 'n'], return_list=False):
    """
    z : np.ndarray (N, latent_dim)
    y : np.ndarray (N,)
    return_list : If True, returns [fig1, fig2, fig3]. If False, returns one large fig.
    """

    cat_dict = get_categories_dict() # {'n': 1, 'g': 0, 'p': 2}
    name_map = get_names()           # {'g': 'γ', 'n': 'n', 'p': 'Pile-up
    colors = {'g': '#007bff', 'n': '#ff7f0e', 'p': '#a9a9a9'}
    alphas = {'g': 0.8,       'n': 0.8,       'p': 0.2}


    # --- Fit 3D PCA on latent space ---
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)
    evr = pca.explained_variance_ratio_

    pairs = [
        (0, 1, "PC1 vs PC2"),
        (0, 2, "PC1 vs PC3"),
        (1, 2, "PC2 vs PC3"),
    ]
    def _plot_axes(ax, ax_idx, ay_idx):
        """Helper to draw points in the correct order on a specific axis."""
        for cat_key in draw_priority:
            class_idx = cat_dict[cat_key]
            mask = (y == class_idx)
            ax.scatter(
                z_pca[mask, ax_idx], 
                z_pca[mask, ay_idx], 
                s=3, 
                alpha=alphas[cat_key], 
                color=colors[cat_key], 
                label=name_map[cat_key]
            )

    if not return_list:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for i, (ax_idx, ay_idx, title) in enumerate(pairs):
            _plot_axes(axs[i], ax_idx, ay_idx)
            axs[i].set_title(f"{title}\n({evr[ax_idx]:.1%} + {evr[ay_idx]:.1%} var)")
            axs[i].set_xlabel(f"PC{ax_idx+1}"); axs[i].set_ylabel(f"PC{ay_idx+1}")
        
        # Consistent legend order: γ, n, Pile-up
        handles, labels = axs[0].get_legend_handles_labels()
        # Re-sort legend to match specific order for the paper
        order = [labels.index(name_map[k]) for k in ['g', 'n', 'p']]
        fig.legend([handles[i] for i in order], [labels[i] for i in order], 
                   loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, markerscale=4)
        plt.tight_layout()
        return fig
    else:
        # --- W&B Logic: List of 3 Individual Figures ---
        individual_figs = []
        for ax_idx, ay_idx, title in pairs:
            fig, ax = plt.subplots(figsize=(6, 5))
            _plot_axes(ax, ax_idx, ay_idx)
            ax.set_title(f"{title} ({evr[ax_idx]:.1%} + {evr[ay_idx]:.1%} var)")
            ax.legend(markerscale=4)
            plt.tight_layout()
            individual_figs.append(fig)
        return individual_figs

def plot_3d_pca_projections(analyzer):
    """
    2D projections of 3D PCA latent space using GMVAEAnalyzer
    """

    out = analyzer._collect(
        return_latent=True,
        return_y=True,
    )

    plot_3d_pca_projections_from_arrays(
        z=out["z"],
        y=out["y_true"],
    )

def plot_3d_pca_from_arrays(z, y, class_names=('γ', 'n', 'p'),):
    """
    z : np.ndarray (N, latent_dim)
    y : np.ndarray (N,)
    """

    # --- 3D PCA ---
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)
    evar = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#007bff', '#ff7f0e', '#a9a9a9']
    opacities = [0.7, 0.7, 0.2]

    for i, name in enumerate(class_names):
        mask = (y == i)
        ax.scatter(
            z_pca[mask, 0],
            z_pca[mask, 1],
            z_pca[mask, 2],
            s=1.5,
            alpha=opacities[i],
            color=colors[i],
            label=name,
        )

    # --- Explained variance box ---
    stats_text = (
        f"Explained Variance\n"
        f"PC1: {evar[0]:.1%}\n"
        f"PC2: {evar[1]:.1%}\n"
        f"PC3: {evar[2]:.1%}\n"
        f"────────────────\n"
        f"Total: {evar.sum():.1%}"
    )

    ax.text2D(
        1.05, 0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.8,
            edgecolor='gray'
        )
    )
    ax.set_box_aspect([1, 1, 1])

    # --- Formatting ---
    ax.set_title("3D PCA Projection (Latent Space)", fontsize=14, pad=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    ax.legend(
        title="Particle Type",
        markerscale=2,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
    )

    plt.tight_layout()
    return fig

def plot_interactive_3d_pca_from_arrays(z, y, class_names=('γ', 'n', 'p'),):
    """
    z : np.ndarray (N, latent_dim)
    y : np.ndarray (N,)
    """

    # --- 3D PCA ---
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)
    evar = pca.explained_variance_ratio_


    colors = ['#007bff', '#ff7f0e', '#a9a9a9']
    opacities = [0.7, 0.7, 0.2]
    # --- Explained variance box ---
    stats_text = (
        f"Explained Variance\n"
        f"PC1: {evar[0]:.1%}\n"
        f"PC2: {evar[1]:.1%}\n"
        f"PC3: {evar[2]:.1%}\n"
        f"────────────────\n"
        f"Total: {evar.sum():.1%}"
    )
    fig = go.Figure()

    for i, name in enumerate(class_names):
        mask = (y == i)
        fig.add_trace(go.Scatter3d(
            x=z_pca[mask, 0],
            y=z_pca[mask, 1],
            z=z_pca[mask, 2],
            name=name,
            mode='markers',
            marker=dict(
                size=1.5,          # Smaller marker size
                color=colors[i],
                opacity=opacities[i]
            )
        ))

    # Update layout for a cleaner look
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",  # Reference the entire figure area
        x=0.02, y=0.98,              # Position: Top-Left
        showarrow=False,
        align="left",
        bgcolor="rgba(30, 30, 30, 0.7)",
        bordercolor="#444",
        borderwidth=1,
        font=dict(size=11, color="white"),
        # Use 'paper' coordinates to keep it outside the rotating cube
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing='constant', x=0.9, y=0.9) # Legend moved to top-right
    )

    fig.show(renderer="browser")

def plot_3d_pca(analyzer, interactive=False):
    """
    3D PCA of GMVAE latent space using analyzer
    interactive=True opens the figure in a browser 
    """

    out = analyzer._collect(
        return_latent=True,
        return_y=True
    )
    if interactive:
        plot_interactive_3d_pca_from_arrays(
            z=out["z"],
            y=out["y_true"],
        )
    else:
        plot_3d_pca_from_arrays(
            z=out["z"],
            y=out["y_true"],
        )

#---------------------------------
#      ROC curce
#---------------------------------

def plot_roc_curve_from_ararys(probs, labels):
    cat_dict = get_categories_dict() # {'n': 1, 'g': 0, 'p': 2}
    name_map = get_names()           # {'g': 'γ', 'n': 'n', 'p': 'Pile-up

    fig =  plt.figure(figsize=(8, 6))
    for cat, i in cat_dict.items():
        # probs[:, i] is the probability of class i
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name_map[cat]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) per Class')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    return fig

def plot_roc_curve(analyzer):
    """
    PCA of GMVAE latent space using analyzer
    """

    out = analyzer._collect(
        return_y=True,
        return_probs=True
    )

    plot_roc_curve_from_ararys(
        probs = out["probs"], 
        labels = out["y_true"]
    )

#---------------------------------
#      Calibration Curves
#---------------------------------
# plots the mean predicted probability against the actual fraction of positives

def plot_calibrationCurve_from_arrays(y_true_cat, prob_cat, cat='p'):
    if (cat!='p') and (cat!='n') and (cat!='g'):
        print('cat must be p, n or g')
        return 0
        
    prob_true, prob_pred = calibration_curve(y_true_cat, prob_cat, n_bins=10 )
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, 'o-', label='GMVAE')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel(f"Predicted P({cat})")
    plt.ylabel(f"Observed fraction {cat}")
    plt.title(f"{cat} confidence calibration")
    plt.legend()

def plot_calibrationCurve(analyzer, cat):
    dict_cats  = get_categories_dict()
    
    out = analyzer._collect(
        return_probs=True,
        return_y=True
    )

    prob_cat = out['probs'][:, dict_cats[cat]] 
    y_true_cat = (out['y_true'] == dict_cats[cat])
    plot_calibrationCurve_from_arrays(y_true_cat, prob_cat, cat)

def plot_all_calibration_curves_from_arrays(y_true, probs):
    dict_cats = get_categories_dict()
    colors = {'g': 'blue', 'n': 'green', 'p': 'red'}
    names  = get_names()

    fig = plt.figure(figsize=(8, 8))
    for cat, idx in dict_cats.items():
        # Prepare arrays for specific category
        prob_cat = probs[:, idx]
        y_true_cat = (y_true == idx)
        
        # Calculate curve
        prob_true, prob_pred = calibration_curve(y_true_cat, prob_cat, n_bins=10)
        
        # Plot individual category
        plt.plot(prob_pred, prob_true, 'o-', label=f'{names[cat]}', color=colors[cat])
    
    # Reference line and formatting
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Fraction")
    plt.title("Calibration Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return fig

def plot_all_calibration_curves(analyzer):
    out = analyzer._collect(return_probs=True, return_y=True)
    plot_all_calibration_curves_from_arrays(out["y_true"], out["probs"])

#---------------------------------
#      Separation Histograms
#---------------------------------
# how well one category is separated from the others
def plot_separation_from_arrays(cat_prob, cat_true, cat, Nbins=30, log=False):
    names = get_names()
    fig = plt.figure(figsize=(8,5))
    plt.hist(cat_prob[cat_true],  bins=Nbins, range=(0., 1.), alpha=0.6, label=f'True {names[cat]}', density=True)
    plt.hist(cat_prob[~cat_true], bins=Nbins, range=(0., 1.), alpha=0.6, label=f'Not {names[cat]}',  density=True)
    plt.xlabel(f"P({names[cat]})")
    plt.ylabel("Normalized")
    plt.legend()
    if log:
        plt.yscale('log')
    plt.title(f"{names[cat]} probability separation")
    return fig

def plot_separation(analyzer, cat, Nbins=30, log=False):
    dict_cats  = get_categories_dict()

    out = analyzer._collect(
        return_probs=True,
        return_y=True
    )
    cat_prob   = out['probs'][:, dict_cats[cat]]
    cat_true = (out['y_true'] == dict_cats[cat])
    plot_separation_from_arrays(cat_prob, cat_true, cat, Nbins, log)

def plot_all_separations(analyzer, Nbins=30, log=False):
    dict_cats = get_categories_dict()
    names = get_names()
    
    # Collect data once to save time
    out = analyzer._collect(return_probs=True, return_y=True)
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (cat_key, idx) in enumerate(dict_cats.items()):
        cat_prob = out['probs'][:, idx]
        cat_true = (out['y_true'] == idx)
        
        ax = axes[i]
        ax.hist(cat_prob[cat_true], bins=Nbins, range=(0., 1.), alpha=0.6, 
                label=f'True {names[cat_key]}', density=True)
        ax.hist(cat_prob[~cat_true], bins=Nbins, range=(0., 1.), alpha=0.6, 
                label=f'Not {names[cat_key]}', density=True)
        
        ax.set_xlabel(f"P({names[cat_key]})")
        ax.set_ylabel("Normalized Density")
        ax.set_title(f"{names[cat_key]} Separation")
        ax.legend()
        
        if log:
            ax.set_yscale('log')
            
    plt.tight_layout()


#---------------------------------
#      logs for wandb
#---------------------------------

import io
from PIL import Image

def fig_to_wandb_image(fig):
    """
    Converts a Matplotlib figure to a W&B-ready image 
    with high DPI and no excess white space.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=120)
    buf.seek(0)
    img = Image.open(buf)
    return wandb.Image(img)


def log_clustering_quality(analyzer, step=None, sample_size=5000):

    # The score is a ratio of cohesion (how close points are to their own cluster) and separation (how far they are from the nearest neighboring cluster). 

#     0.71 – 1.0 (Strong): You have very clear, dense clusters. Your pulses are being perfectly discriminated.
#     0.51 – 0.70 (Reasonable): Good separation, but some pulses might be "borderline" (common with pile-up events that look like neutrons).
#     0.26 – 0.50 (Weak): Clusters are loose and likely overlapping. This usually correlates with the "flat" plateau in your calibration curves.
#     < 0.25 (Poor/No Clustering): The latent space is a "hairball." The model hasn't learned distinct features for neutrons vs. gammas


    # Collect latent vectors (mu) and true labels
    out = analyzer._collect(return_y=True, return_latent=True)
    z = out["z"]        # High-dimensional latent space
    labels = out["y_true"]

    # Calculate mean Silhouette Score
    # Range: -1 (wrongly clustered) to +1 (perfectly separated)
    if sample_size>0:
        score = silhouette_score(z, labels, sample_size=sample_size, random_state=42)
    else:
        score = silhouette_score(z, labels)
    
    # Log to W&B
    wandb.log({"Clustering/Silhouette_Score": score,
               "epoch": step  })
    return score


def run_final_inference_report(analyzer):
    """
    Performs full inference, generates all diagnostic plots, 
    and logs them to the 'Inference/' namespace in W&B.
    """
    print("🚀 Starting Final Inference Evaluation...")
    
    # 1. Collect everything in one go (efficient)
    out = analyzer._collect(
        return_x=True,
        return_y=True, 
        return_probs=True, 
        return_preds=True, 
        return_latent=True,
        return_reco=True
    )
    x = out["x"]
    x_hat = out["x_hat"]
    y_true = out["y_true"]
    y_pred = out["preds"]
    probs = out["probs"]
    z = out["z"]
    
    report_dict = {}

    # --- Confusion Matrix ---
    fig_cm = plot_confusion_matrix_from_arrays(y_true, y_pred)
    report_dict["Inference_General/Confusion_Matrix"] = fig_to_wandb_image(fig_cm)
    plt.close(fig_cm)

    # --- Calibration Curves ---
    fig_cal = plot_all_calibration_curves_from_arrays(y_true, probs)
    report_dict["Inference_General/Calibration"] = fig_to_wandb_image(fig_cal)
    plt.close(fig_cal)

    # --- ROC Curves ---
    fig_roc = plot_roc_curve_from_ararys(probs, y_true)
    report_dict["Inference_General/ROC"] = fig_to_wandb_image(fig_roc)
    plt.close(fig_roc)

    # --- Reconstruction ---
    fig_reco = plot_random_reconstructions_from_arrays(x, x_hat, y_true, y_pred, num_samples=6, class_names=('γ', 'n', 'p'), seed=42)
    report_dict["Inference_Reconstruction/ROC"] = fig_to_wandb_image(fig_reco)
    plt.close(fig_reco)

    # --- Separation Histograms  ---
    dict_cats  = get_categories_dict() # {'g': 0, 'n': 1, 'p': 2}
    name = get_names() # {'g': 'γ', 'n': 'n', 'p': 'Pile-up'}
    for k, v in dict_cats.items():
        cat_prob   = out['probs'][:, v]
        cat_true = (out['y_true'] == v)
        fig_sep = plot_separation_from_arrays(cat_prob, cat_true, k, Nbins=50, log=True)
        report_dict[f"Inference_separation/Separation_{name[k]}"] = fig_to_wandb_image(fig_sep)
        plt.close(fig_sep)

    # --- PCA  ---
    pca_figs = plot_3d_pca_projections_from_arrays(z, y_true, return_list=True, draw_priority = ['p', 'g', 'n'])
    wandb.log({
    "Inference_Latent/PCA_PC1_PC2": fig_to_wandb_image(pca_figs[0]),
    "Inference_Latent/PCA_PC1_PC3": fig_to_wandb_image(pca_figs[1]),
    "Inference_Latent/PCA_PC2_PC3": fig_to_wandb_image(pca_figs[2]),
    })
    for f in pca_figs: plt.close(f)

    pca_3d = plot_3d_pca_from_arrays(z, y_true, class_names=('γ', 'n', 'p'))
    report_dict[f"Inference_Latent/PCA_3D"] = fig_to_wandb_image(pca_3d)
    plt.close(pca_3d)

    # --- t-SNE ---
    cat_dict = get_categories_dict()
    name_map = get_names()
    colors = {0: '#007bff', 1: '#ff7f0e', 2: '#a9a9a9'} 
    # Sampling for speed if dataset is huge, but keeping it representative
    Nsamples = 10000
    idx = np.random.choice(len(z), min(len(z), Nsamples), replace=False)
    z_sub, y_sub = z[idx], y_true[idx]
    tsne_res = TSNE(n_components=2, random_state=42).fit_transform(z_sub)

    fig_lat, ax = plt.subplots(figsize=(7, 6))
    for cat_key, class_idx in cat_dict.items():
        mask = (y_sub == class_idx)
        ax.scatter(
            tsne_res[mask, 0], 
            tsne_res[mask, 1], 
            c=colors[class_idx], 
            label=name_map[cat_key],
            alpha=0.6, 
            s=3,
            # Plot pile-up first (zorder=1) so it stays in background
            zorder=1 if cat_key == 'p' else 2 
        )
    ax.set_title("t-SNE Latent Space Comparison")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Particle Type", markerscale=4, loc='best')
    report_dict["Inference_Latent/t-SNE"] = fig_to_wandb_image(fig_lat)
    plt.close(fig_lat)





    # --- 5. Log Summary Metrics ---
    # These show up in the W&B run table (not a plot)
    score = silhouette_score(z, y_true, sample_size=Nsamples, random_state=42)
    wandb.run.summary["final_silhouette_score"] = score

    # Push all plots to W&B
    wandb.log(report_dict)
    print("✅ Final Report Sent to W&B.")
