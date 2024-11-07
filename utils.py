import numpy as np
import torch

import matplotlib.pyplot as plt

from typing import List

def random_data_visual(dataset: torch.utils.data.Dataset,
                       class_names: List) -> None:
    """
    A function to visualize random 9 datapoints from the FashionMNIST library.
    
    Args:
        dataset: A dataset from which 9 random samples are viewed.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 3, 3
    
    for i in range(rows * cols):
        # Getting an index for the random sample
        random_index = torch.randint(low = 0, high = len(dataset), size=[1]).item()
        
        # Getting the data from the random sample
        image = dataset[random_index][0].permute(1,2,0).numpy()
        label = dataset[random_index][1]
        
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(image, 'gray')
        plt.title(class_names[label])
        plt.axis(False)