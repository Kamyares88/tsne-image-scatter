import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

def points_overlap(p1, p2, min_dist):
    """Check if two points are too close to each other"""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)) < min_dist

def adjust_point_position(point, existing_points, min_dist, max_attempts=50):
    """
    Adjust point position if it overlaps with existing points.
    Uses a spiral pattern to find new positions.
    """
    if not existing_points:
        return point

    original_point = np.array(point)
    current_point = original_point.copy()
    
    # Check if the original position is fine
    overlap = any(points_overlap(current_point, p, min_dist) for p in existing_points)
    if not overlap:
        return current_point

    # If there's overlap, try to find a new position
    theta = 0
    radius = min_dist
    for _ in range(max_attempts):
        # Move in a spiral pattern
        theta += 0.5
        radius += min_dist / 10
        
        # Calculate new position
        current_point[0] = original_point[0] + radius * np.cos(theta)
        current_point[1] = original_point[1] + radius * np.sin(theta)
        
        # Check if new position is good
        if not any(points_overlap(current_point, p, min_dist) for p in existing_points):
            return current_point

    # If we couldn't find a non-overlapping position, return the original point
    return original_point

def imscatter(x, y, images, ax=None, zoom=1.0, min_dist=None):
    if ax is None:
        ax = plt.gca()
    
    if min_dist is None:
        # Estimate minimum distance based on figure size and zoom
        bbox = ax.get_window_extent()
        min_dist = zoom * 100 * (bbox.width / len(x))  # Adjust this multiplier as needed
    
    artists = []
    placed_points = []
    
    # Sort points by distance from center to handle central clustering better
    points = np.column_stack([x, y])
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    sorted_indices = np.argsort(distances)
    
    for idx in sorted_indices:
        point = points[idx]
        img = images[idx]
        
        # Adjust position to avoid overlap
        adjusted_point = adjust_point_position(point, placed_points, min_dist)
        placed_points.append(adjusted_point)
        
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, adjusted_point, xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def visualize_tsne_images(tsne_results, images, n_samples=1000, figsize=(20, 20), zoom=0.3):
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
    zoom : float
        Zoom factor for the images
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
    imscatter(tsne_subset[:, 0], tsne_subset[:, 1], images_subset, ax=ax, zoom=zoom)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE Visualization with Images (n={len(tsne_subset)})')
    
    return fig

# Example usage:
"""
# Assuming you have:
# tsne_results: numpy array of shape (8000, 2)
# images: list of 8000 images (as numpy arrays)

fig = visualize_tsne_images(tsne_results, images, n_samples=1000, zoom=0.3)
plt.show()
""" 