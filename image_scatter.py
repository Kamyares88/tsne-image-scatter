import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

def imscatter(x, y, images, ax=None, zoom=1.0):
    if ax is None:
        ax = plt.gca()
    
    artists = []
    for x0, y0, img in zip(x, y, images):
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def visualize_tsne_images(tsne_results, images, n_samples=1000, figsize=(20, 20)):
    """
    Visualize t-SNE results with images
    
    Parameters:
    -----------
    tsne_results : numpy array
        Array of shape (n_samples, 2) containing t-SNE coordinates
    images : list or numpy array
        List of images corresponding to each point
    n_samples : int
        Number of samples to plot (for downsampling)
    figsize : tuple
        Size of the output figure
    """
    # Downsample if necessary
    if len(tsne_results) > n_samples:
        indices = random.sample(range(len(tsne_results)), n_samples)
        tsne_subset = tsne_results[indices]
        images_subset = [images[i] for i in indices]
    else:
        tsne_subset = tsne_results
        images_subset = images

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot images instead of points
    imscatter(tsne_subset[:, 0], tsne_subset[:, 1], images_subset, ax=ax, zoom=0.3)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE Visualization with Images (n={len(tsne_subset)})')
    
    return fig

# Example usage:
"""
# Assuming you have:
# tsne_results: numpy array of shape (8000, 2)
# images: list of 8000 images (as numpy arrays)

fig = visualize_tsne_images(tsne_results, images, n_samples=1000)
plt.show()
""" 