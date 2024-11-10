import os
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

def random_data_visual(dataset: torch.utils.data.Dataset[Tuple[torch.Tensor, int]],
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

def plot_save_loss_curves(results: Dict[str, List[float]],
                          model_name: str) -> None:
    """
    Plots and saves the training and test loss curves.

    This function takes the results of a model's training process, which includes 
    the training and test losses, and plots them against the number of epochs. 
    The plot is then saved as a PNG file in the './results' directory with the 
    model's name as the filename.

    Args:
        results (Dict[str, List[float]]): A dictionary containing 'train_loss' 
                                          and 'test_loss' lists for each epoch.
        model_name (str): The name of the model, used for the plot title and filename.

    Returns:
        None: The function does not return any value; it only saves the plot.
    """
    # Creating the data path
    results_path = Path('./results')
    if not results_path.is_dir():
        os.mkdir(results_path)
        
    # Get loss lists
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    
    epochs = range(len(results['train_loss']))
    
    # Plotting
    plt.figure(figsize = (6,4))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title(f'{model_name}: Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Saving the figure
    model_name = model_name + '.png'
    save_path = results_path / model_name
    
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()

def save_model(model: torch.nn.Module,
               model_name: str):
    """
    Saves the model's state_dict to a specified directory.

    This function creates the target directory (if it doesn't exist), checks that the 
    model name ends with '.pth' or '.pt', and saves the model's state_dict to the 
    specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str): The directory where the model will be saved.
        model_name (str): The name of the model file (should end with '.pth' or '.pt').

    Raises:
        AssertionError: If the `model_name` does not end with '.pth' or '.pt'.
    
    Returns:
        None: The function saves the model and does not return anything.
    """
    # Creating the save path
    models_path = Path('./models')
    if not models_path.is_dir():
        os.mkdir(models_path)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "The argument 'model_name' should end with '.pt' or '.pth'"
    model_save_path = models_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)