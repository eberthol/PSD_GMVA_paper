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

# !pip3 install umap-learn
# import umap

#!pip3 install plotly
import plotly.graph_objects as go # interactive plots

# synthData_path = os.path.join('..', 'synthetic_data') 
# sys.path.append(synthData_path)
# import generate_synthetic_data as gsd

# import gmvae_architecture as ga



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
def plot_confusion_matrix(dataloader, model):
    """
    Plot normalized confusion matrix (values in %)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits, _, _ = model.encoder(x)  # could use the full model instead, but it returns the same values [x_hat, logits, mu, logvar = model(x) -> logits are the same]
            preds = torch.argmax(logits, dim=1)

            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=['γ','n','p'])
    disp.plot(cmap=plt.cm.Blues, values_format='.1%')
    plt.title("Normalized confusion matrix")
    plt.show()

#---------------------------------
#      Reconstruction
#---------------------------------

def reconstruction_error(model, x_sample, y_sample):
    """
    Get total reconstruction error and errors for each categories
    x_sample, y_sample: can the full dataset or a subsample of it
    """
    model.eval()
    with torch.no_grad():
        x_hat, _, _, _  = model(x_sample)
    # Calculate overall error (aka reconstruction error)
    overall_mse = torch.mean((x_sample - x_hat)**2).item()
    print(f"Overall MSE: {overall_mse:.6f}")

    # Calculate error PER CATEGORY 
    for label, name in enumerate(['Neutron', 'Gamma', 'Pile-up']):
        mask = (y_sample == label)
        if mask.sum() > 0:
            cat_mse = torch.mean((x_sample[mask] - x_hat[mask])**2).item()
            print(f" {name} Reconstruction Error: {cat_mse:.6f}")

def plot_random_reconstructions(model, dataloader, num_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    class_names = ['γ', 'n', 'p']
    
    with torch.no_grad():
        x_batch, y_true_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)
        
        # Get reconstructions and predictions
        x_hat_batch, logits, _, _ = model(x_batch)
        y_pred_batch = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Convert to numpy
        x_batch = x_batch.cpu().numpy()
        x_hat_batch = x_hat_batch.cpu().numpy()
        y_true_batch = y_true_batch.numpy()

    # Pick samples
    idxs = np.random.choice(len(x_batch), num_samples, replace=False)

    _, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i, idx in enumerate(idxs):
        ax = axes[i]
        ax.plot(x_batch[idx], label='Original', color='#1f77b4', linewidth=1.5)
        ax.plot(x_hat_batch[idx], '--', label='Reconstructed', color='#ff7f0e', linewidth=1.5)
        
        # Formatting
        title_color = 'darkgreen' if y_true_batch[idx] == y_pred_batch[idx] else 'darkred'
        ax.set_title(f"True: {class_names[y_true_batch[idx]]} | Pred: {class_names[y_pred_batch[idx]]}", 
                     fontsize=12, fontweight='bold', color=title_color)
        
        ax.set_xlabel("Time (Samples)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0: # Only add legend to the first plot to save space
            ax.legend(loc='upper right')

    plt.tight_layout()

#---------------------------------
#      Latent Space
#---------------------------------
def plot_2d_pca(dataloader, model, device='cpu'):

    features, labels = [], []
    model.eval()
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # Use encoder to get high-dimensional latent vectors
            logits, _, _ = model.encoder(x) 
            
            features.append(logits.cpu().numpy())
            labels.append(y.numpy())

    # Flatten collected batches into single arrays
    X = np.vstack(features)
    y = np.hstack(labels)

    
    # --- 2D PCA ---
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    evar = pca_2d.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot classes with small markers and custom labels
    class_names = ['γ', 'n', 'p']
    colors = ['#007bff', '#ff7f0e', '#a9a9a9']
    opacities = [0.7, 0.7, 0.2]

    for i, name in enumerate(class_names):
        mask = (y == i)
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   label=name, s=2, alpha=opacities[i], color=colors[i])

    # Add the Stats Text Box
    stats_text = (
        f"Explained Variance\n"
        f"PC1: {evar[0]:.1%}\n"
        f"PC2: {evar[1]:.1%}\n"
        f"────────────────\n"
        f"Total: {sum(evar):.1%}"
    )

    # Place text in the top-left corner (0.02, 0.98) relative to axes
    ax.text(1.05, 0.65, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Final Formatting
    ax.set_title("2D PCA Projection", fontsize=14, pad=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Place legend outside to the right
    ax.legend(title="Particle Type", markerscale=5, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()

def plot_interactive_3d_pca(dataloader, model, device='cpu'):
    features, labels = [], []
    model.eval()
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # Use encoder to get high-dimensional latent vectors
            logits, _, _ = model.encoder(x) 
            features.append(logits.cpu().numpy())
            labels.append(y.numpy())

    X = np.vstack(features)
    y = np.hstack(labels)
    
    # PCA to 3 dimensions
    pca_3d = PCA(n_components=3)
    X_pca = pca_3d.fit_transform(X)


    evar = pca_3d.explained_variance_ratio_
    stats_text = (
        f"<b>Explained Variance</b><br>"
        f"PC1: {evar[0]:.1%}<br>"
        f"PC2: {evar[1]:.1%}<br>"
        f"PC3: {evar[2]:.1%}<br>"
        f"──────────────<br>"
        f"<b>Total: {sum(evar):.1%}</b>"
    )
    
    class_names = ['γ', 'n', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#a9a9a9'] # Blue, Orange, Gray
    opacities = [0.8, 0.8, 0.2] 

    fig = go.Figure()

    for i, name in enumerate(class_names):
        mask = (y == i)
        fig.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
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

    # fig.show()
    fig.show(renderer="browser")
