import torch

class LinearFmnist(torch.nn.Module):
    """
    A simple fully-connected neural network for classification tasks like FashionMNIST.

    The network consists of four linear layers: three hidden layers with the same 
    number of hidden units and one output layer for classification. The architecture 
    is defined using `torch.nn.Sequential`.

    Args:
        input_units (int): Number of input features (e.g., 784 for flattened 28x28 images).
        output_units (int): Number of output units (e.g., 10 for 10 classes).
        hidden_units (int): Number of units in each hidden layer.

    Forward pass:
        x (torch.Tensor): Input tensor with shape (batch_size, input_units).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_units).
    """
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hidden_units: int) -> None:
        super().__init__()
        
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_units,
                            out_features=hidden_units),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=hidden_units),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=hidden_units),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=output_units)
        )
        
    def forward(self, x):
        return self.sequence(x)

class NonLinearFmnist(torch.nn.Module):
    """
    A neural network for classification tasks with non-linear activation functions.

    This network consists of four fully connected layers with ReLU activations in 
    the first two hidden layers and a Sigmoid activation in the third hidden layer. 
    It is designed for tasks like FashionMNIST, where the input is typically a flattened 
    image and the output corresponds to class scores.

    Args:
        input_units (int): Number of input features (e.g., 784 for flattened 28x28 images).
        output_units (int): Number of output units (e.g., 10 for 10 classes).
        hidden_units (int): Number of units in each hidden layer.

    Forward pass:
        x (torch.Tensor): Input tensor with shape (batch_size, input_units).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_units).
    """
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hidden_units: int) -> None:
        super().__init__()
        
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_units,
                            out_features=hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=hidden_units),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=hidden_units,
                            out_features=output_units)
        )
        
    def forward(self, x):
        return self.sequence(x)

class ConvolutionFmnist(torch.nn.Module):
    """
    Convolutional neural network for image classification using TinyVGG architecture.

    The network consists of two convolutional blocks with ReLU activations and max pooling, 
    followed by a fully connected classifier for predicting class labels.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale).
        out_features (int): Number of output units (e.g., 10 for FashionMNIST classes).
        hidden_channels (int): Number of channels in hidden layers.
        image_height (int): Height of input image (e.g., 28 for FashionMNIST).
        image_width (int): Width of input image (e.g., 28 for FashionMNIST).

    Forward pass:
        x (torch.Tensor): Input tensor with shape (batch_size, in_channels, image_height, image_width).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
    """
    def __init__(self,
                 in_channels: int,
                 out_features: int,
                 hidden_channels: int,
                 image_height: int,
                 image_width: int) -> None:
        super().__init__()
        
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_channels * int((image_height/2)/2) * int((image_width/2)/2),
                            out_features=out_features)
        )
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x