# Experiments with Fashion-MNIST Dataset

This repository is focused on training different models on Fashion-MNIST dataset. There will be three different models which will trained, tested and compared on this dataset. The three models chosen will be the three different deep learning techniques a beginner would learn at the start of their 'deep-learning-journey' and this is becasue Fashion-MNIST is a beginner dataset.

The three models for experimentation:
- **Linear Model**: This model will just have dense linear layers with no activation functions.
- **Non-Linear Model**: This model will combine the dense linear layers with non-linear activation functions to provide the model with the ability to learn non-linear patterns in data.
- **Convolutional Model**: This model will use a convolutional neural network with pooling. This model will replicate the Tiny-VGG model that has been visually demonstrated [here]('https://poloclub.github.io/cnn-explainer/').

## Requirements

- Ubuntu (Only tested on Ubuntu)
- PyTorch >= 2.5.1
- Torchvision >= 0.20.1
- Torchinfo >= 1.8.0
- Torchmetrics >= 1.5.2

```
pip install -r requirements.txt
```

## Usage

Train all three models by running the `train.py` script. There are hyperparameters you can change within the training script.

```
python train.py
```

## Results

    Models trained on:
    - Epochs: 15
    - Learning Rate: 1e-3
    - Batch Size: 32

All the models approximately had same number of parameters which is ~13,000. The convolutional model was the best performing model out of all three whereas the second best model was the linear model. The model with the non-linear activations had the worst performance.

### Accuracy on test data:
- Linear Model: 79% 
- Non-Linear Model: 45%
- Convolutional Model: 83%

### Loss Curves

The loss values after the training:
- Linear Model: 
    - Training: 0.57
    - Testing: 0.59
- Non-Linear Model: 
    - Training: 2.03
    - Testing: 1.97
- Convolutional Model: 
    - Training: 0.45
    - Testing: 0.47

---

#### Linear Model

![Loss curves of Linear Model](./results/LinearFmnist.png)

---

#### Non-Linear Model

![Loss curves of Non-Linear Model](./results/NonLinearFmnist.png)

---

#### Convolution Model

![Loss curves of Convolution Model](./results/ConvolutionFmnist.png)

---