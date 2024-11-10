import torch

import torchmetrics

import data_setup, model_builder, engine, utils

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    models = []
    optimizers = []
    results = []
    
    # Device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Downloading the data
    train_dataloader, test_dataloader, class_names = data_setup.get_dataloaders(batch_size=BATCH_SIZE,
                                                                                num_workers=NUM_WORKERS)
    
    # Creating the three models
        # Hidden units for linear and non-linear models is 16
        # Hidden channels for convolutional model is 15
        # To have similar number of parameters for all three models ~13,000
    model_linear = model_builder.LinearFmnist(input_units=28 * 28,                                              # Image height and width
                                              output_units=len(class_names),
                                              hidden_units=16)
    
    model_non_linear = model_builder.NonLinearFmnist(input_units=28 * 28,
                                                     output_units=len(class_names),
                                                     hidden_units=16)
    
    model_convolutional = model_builder.ConvolutionFmnist(in_channels=1,
                                                          out_features=len(class_names),
                                                          hidden_channels=15)
    
    # Adding the models to the list
    models.append(model_linear)
    models.append(model_non_linear)
    models.append(model_convolutional)
    
    # Loss, optimizer and accuracy
    loss_function = torch.nn.CrossEntropyLoss()
    accuracy_function = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names))
    
    for model in models:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                   lr = 0.001)
        optimizers.append(optimizer)
    
    # Training the models
    for model, optimizer in zip(models, optimizers):
        result = engine.train(epochs = 10,
                              model=model,
                              train_dataloader=train_dataloader,
                              test_dataloader=test_dataloader,
                              loss_function=loss_function,
                              optimizer=optimizer,
                              accuracy_function=accuracy_function,
                              device = device)
        
        # Appending the training result in results list
        results.append(result)