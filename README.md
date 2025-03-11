# t-SNE Image Scatter Plot

A Python tool for visualizing t-SNE embeddings using images instead of points in a scatter plot. This is particularly useful for visualizing high-dimensional data where each point is associated with an image.

## Features

- Create scatter plots with images instead of points
- Automatic downsampling for large datasets
- Configurable image size and figure dimensions
- Easy-to-use interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from image_scatter import visualize_tsne_images

# Prepare your data
tsne_results  # Your t-SNE coordinates of shape (n_samples, 2)
images        # Your list of images as numpy arrays

# Create the visualization
fig = visualize_tsne_images(tsne_results, images, n_samples=1000)
plt.show()
```

## Parameters

- `tsne_results`: numpy array of shape (n_samples, 2) containing t-SNE coordinates
- `images`: list of images (as numpy arrays) corresponding to each point
- `n_samples`: number of samples to plot (for downsampling)
- `figsize`: size of the output figure (default: (20, 20))

## Requirements

- numpy
- matplotlib

## License

MIT License 