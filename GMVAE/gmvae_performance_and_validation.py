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

# !pip3 install umap-learn
# import umap

#!pip3 install plotly
import plotly.graph_objects as go # interactive plots


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
    plt.show()

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
    plt.show()

#---------------------------------
#      Confusion Matrix
#---------------------------------

def plot_confusion_matrix_from_arrays(y_true, y_pred, class_names=('γ', 'n', 'p'), normalize=True, title="Normalized Confusion Matrix"):
    """
    y_true : np.ndarray (N,)
    y_pred : np.ndarray (N,)
    """

    norm = "true" if normalize else None

    cm = confusion_matrix(y_true, y_pred, normalize=norm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".1%" if normalize else "d", colorbar=True)

    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

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

def plot_random_reconstructions_from_arrays(
    x,
    x_hat,
    y_true,
    y_pred,
    num_samples=6,
    class_names=('γ', 'n', 'p'),
    seed=None,
):
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
    plt.show()

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
    plt.show()

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

def plot_3d_pca_projections_from_arrays(z, y, class_names=('γ', 'n', 'p'),):
    """
    z : np.ndarray (N, latent_dim)
    y : np.ndarray (N,)
    """

    # --- Fit 3D PCA on latent space ---
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)
    evr = pca.explained_variance_ratio_

    # --- Create figure ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#007bff', '#ff7f0e', '#a9a9a9']
    opacities = [0.7, 0.7, 0.2]

    pairs = [
        (0, 1, "PC1 vs PC2"),
        (0, 2, "PC1 vs PC3"),
        (1, 2, "PC2 vs PC3"),
    ]

    for i, (ax_idx, ay_idx, title) in enumerate(pairs):
        ax = axs[i]

        for c_idx, name in enumerate(class_names):
            mask = (y == c_idx)
            ax.scatter(
                z_pca[mask, ax_idx],
                z_pca[mask, ay_idx],
                s=3,
                alpha=opacities[c_idx],
                color=colors[c_idx],
                label=name,
            )

        ax.set_title(
            f"{title}\n({evr[ax_idx]:.1%} + {evr[ay_idx]:.1%} var)",
            fontsize=12,
        )
        ax.set_xlabel(f"PC{ax_idx + 1}")
        ax.set_ylabel(f"PC{ay_idx + 1}")
        ax.grid(True, linestyle="--", alpha=0.3)

    # --- Shared legend ---
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(class_names),
        title="Particle Type",
        markerscale=4,
    )

    plt.tight_layout()
    plt.show()

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

    #  class_names = ['γ', 'n', 'p']
    # colors = ['#1f77b4', '#ff7f0e', '#a9a9a9'] # Blue, Orange, Gray
    # opacities = [0.8, 0.8, 0.2] 

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
    plt.show()

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

def plot_roc_curve_from_ararys(probs, labels, class_names=['γ', 'n', 'p']):
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        # probs[:, i] is the probability of class i
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) per Class')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

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
    plt.show()

def plot_calibrationCurve(analyzer, cat):
    dict_cats  = get_categories_dict()
    
    out = analyzer._collect(
        return_probs=True,
        return_y=True
    )

    prob_cat = out['probs'][:, dict_cats[cat]] 
    y_true_cat = (out['y_true'] == dict_cats[cat])
    plot_calibrationCurve_from_arrays(y_true_cat, prob_cat, cat)

def plot_all_calibration_curves(analyzer):
    dict_cats = get_categories_dict()
    colors = {'g': 'blue', 'n': 'green', 'p': 'red'}
    names  = get_names()
    
    out = analyzer._collect(return_probs=True, return_y=True)
    
    plt.figure(figsize=(8, 8))
    
    for cat, idx in dict_cats.items():
        # Prepare arrays for specific category
        prob_cat = out['probs'][:, idx]
        y_true_cat = (out['y_true'] == idx)
        
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
    plt.show()



#---------------------------------
#      Separation plots
#---------------------------------
# how well one category is separated from the others
def plot_separation_from_arrays(cat_prob, cat_true, cat, Nbins=30, log=False):
    names = get_names()

    plt.figure(figsize=(8,5))
    plt.hist(cat_prob[cat_true],  bins=Nbins, range=(0., 1.), alpha=0.6, label=f'True {names[cat]}', density=True)
    plt.hist(cat_prob[~cat_true], bins=Nbins, range=(0., 1.), alpha=0.6, label=f'Not {names[cat]}',  density=True)
    plt.xlabel(f"P({names[cat]})")
    plt.ylabel("Normalized")
    plt.legend()
    if log:
        plt.yscale('log')
    plt.title(f"{names[cat]} probability separation")
    plt.show()

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
    plt.show()
