import torch

class LinearFmnist(torch.nn.Module):
    """
    
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
    Architecture: TinyVGG
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