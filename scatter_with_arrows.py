import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import cKDTree

def find_label_position(point, existing_positions, ax_lims, min_dist=0.1):
    """Find a suitable position for the label that doesn't overlap with others"""
    x_min, x_max = ax_lims[0]
    y_min, y_max = ax_lims[1]
    
    # Create a grid of possible positions around the point
    theta = np.linspace(0, 2*np.pi, 16)  # 16 possible angles
    r = np.linspace(min_dist, min_dist*2, 3)  # 3 possible distances
    
    best_pos = None
    best_dist = -np.inf
    
    for radius in r:
        for angle in theta:
            # Calculate potential position
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            new_pos = [point[0] + dx, point[1] + dy]
            
            # Check if position is within plot limits
            if (new_pos[0] < x_min or new_pos[0] > x_max or 
                new_pos[1] < y_min or new_pos[1] > y_max):
                continue
            
            # If no existing positions, use this one
            if len(existing_positions) == 0:
                return new_pos
            
            # Calculate minimum distance to existing positions
            min_dist_to_existing = np.min([
                np.sqrt((p[0]-new_pos[0])**2 + (p[1]-new_pos[1])**2)
                for p in existing_positions
            ])
            
            # Update best position if this is better
            if min_dist_to_existing > best_dist:
                best_dist = min_dist_to_existing
                best_pos = new_pos
    
    # If no position found, slightly extend the plot limits
    if best_pos is None:
        return [point[0] + min_dist, point[1] + min_dist]
    
    return best_pos

def scatter_with_row_arrows(x, y, figsize=(10, 10), point_size=50, 
                          arrow_color='black', text_color='black',
                          min_label_dist=0.1):
    """
    Create a scatter plot with arrows pointing to row numbers
    
    Parameters:
    -----------
    x, y : array-like
        Coordinates for scatter plot points
    figsize : tuple
        Figure size
    point_size : int
        Size of scatter plot points
    arrow_color : str
        Color of the arrows
    text_color : str
        Color of the row number labels
    min_label_dist : float
        Minimum distance between labels
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(x, y, s=point_size)
    
    # Get plot limits with some padding
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Store label positions
    label_positions = []
    
    # Add arrows and labels for each point
    for i, (xi, yi) in enumerate(zip(x, y)):
        # Find position for label
        label_pos = find_label_position(
            [xi, yi], 
            label_positions,
            [ax.get_xlim(), ax.get_ylim()],
            min_label_dist
        )
        label_positions.append(label_pos)
        
        # Add arrow
        arrow = FancyArrowPatch(
            (xi, yi),
            label_pos,
            arrowstyle='-|>',
            color=arrow_color,
            mutation_scale=10
        )
        ax.add_patch(arrow)
        
        # Add row number label
        ax.text(
            label_pos[0], label_pos[1],
            f' {i}',  # Row number
            color=text_color,
            va='center',
            ha='left'
        )
    
    plt.tight_layout()
    return fig, ax

# Example usage:
"""
# Generate some example data
np.random.seed(42)
x = np.random.randn(20)
y = np.random.randn(20)

# Create the plot
fig, ax = scatter_with_row_arrows(x, y)
plt.show()
""" 