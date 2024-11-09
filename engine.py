import torch

import torchmetrics

from tqdm.auto import tqdm
from typing import Dict, List

def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """
    Trains the model for one epoch and returns the average training loss.

    Performs forward pass, calculates loss, backpropagates gradients, and updates model weights.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for weight updates.
        device (torch.device): The device (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary with the average 'train_loss' for the epoch.
    """
    # Set the model to train mode
    model = model.train()
    
    # Move the model to device
    model = model.to(device)
    
    # Variable to track loss
    train_loss = 0
    
    for X, y in dataloader:
        # Move data to the device
        X, y = X.to(device), y.to(device)
        
        # Forward Pass -> Loss -> Zero Grad -> Back Prop -> Gradient Descent
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Adding the loss
        train_loss += loss
    
    # Average the loss
    train_loss /= len(dataloader)
    
    return {'train_loss': train_loss}

def test_epoch(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_funciton: torch.nn.Module,
               accuracy_function: torchmetrics.Accuracy,
               device: torch.device) -> Dict[str, float]:
    """
    Evaluates the model on the test data and returns the average loss and accuracy.

    This function runs a forward pass through the model for each batch in the test set, 
    calculates the loss and accuracy, and averages them over the entire dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The test dataset.
        loss_function (torch.nn.Module): The loss function to compute the error.
        accuracy_function (torchmetrics.Accuracy): The function to calculate accuracy.
        device (torch.device): The device (CPU or GPU).

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'test_loss' (float): The average loss for the test set.
            - 'test_accuracy' (float): The average accuracy for the test set.
    """
    # Set the model to evaluation mode
    model = model.eval()
    
    # Move the model to the device
    model = model.to(device)
    
    # Tracking variables
    test_loss = 0
    test_accuracy = 0
    
    for X, y in dataloader:
        # Move data to the device
        X, y = X.to(device), y.to(device)
        
        # Forward pass -> Loss
        y_pred = model(X)
        loss = loss_funciton(y_pred, y)
        
        # Calculate the accuracy
        accuracy += accuracy_function(torch.argmax(y_pred, dim=1).squeeze(), y)
        
        # Adding the tracking values
        test_loss += loss
        test_accuracy += accuracy
        
    # Average
    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    
    return {'test_loss': test_loss,
            'test_accuracy': test_accuracy}

def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_function: torchmetrics.Accuracy,
          device: torch.device) -> Dict[str, List[float]]:
    """
    Trains the model for a specified number of epochs and tracks training loss, 
    test loss, and test accuracy.

    This function runs the training and evaluation process for the given number of epochs, 
    and collects metrics such as loss and accuracy for both the training and test sets.

    Args:
        epochs (int): The number of epochs to train the model.
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        test_dataloader (torch.utils.data.DataLoader): The test data loader.
        loss_function (torch.nn.Module): The loss function for training.
        optimizer (torch.optim.Optimizer): The optimizer for model parameter updates.
        accuracy_function (torchmetrics.Accuracy): The function to compute accuracy.
        device (torch.device): The device (CPU or GPU) for training.

    Returns:
        Dict[str, List[float]]: A dictionary containing:
            - 'train_loss' (List[float]): The list of average training losses per epoch.
            - 'test_loss' (List[float]): The list of average test losses per epoch.
            - 'test_accuracy' (List[float]): The list of average test accuracies per epoch.
    """
    # Tracking list
    train_loss = []
    test_loss = []
    test_accuracy = []
    
    for epoch in tqdm(range(epochs)):
        print(f"\nEPOCH: {epoch} ------------------------------------------------------------------------\n")
        train_result = train_epoch(model=model,
                                   dataloader=train_dataloader,
                                   loss_function=loss_function,
                                   optimizer=optimizer,
                                   device=device)
        test_result = test_epoch(model=model,
                                 dataloader=test_dataloader,
                                 loss_funciton=loss_function,
                                 accuracy_function=accuracy_function,
                                 device=device)
        
        # Append to the tracking lists
        train_loss.append(train_result['train_loss'])
        test_loss.append(test_result['test_loss'])
        test_accuracy.append(test_result['test_accuracy'])
        
    return {'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy}
    
    