"""
Utility functions used in Projects 1 and 2
"""
import torch
import torchvision
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import random
import math
import os

#Original Author: Justin C. Johnson https://web.eecs.umich.edu/~justincj/
#Modified: Jonathan Gryak, Spring 2025

################# Utility Functions #################
def hello_helper():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from utils.py!")

def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr
def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """
    Compute the numeric gradient of f at x using a finite differences
    approximation. We use the centered difference:

    df    f(x + h) - f(x - h)
    -- ~= -------------------
    dx           2 * h

    Function can also expand this easily to intermediate layers using the
    chain rule:

    dL   df   dL
    -- = -- * --
    dx   dx   df

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to compute the gradient
    - dLdf: optional upstream gradient for intermediate layers
    - h: epsilon used in the finite difference calculation
    Returns:
    - grad: A tensor of the same shape as x giving the gradient of f at x
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad


def rel_error(x, y, eps=1e-10):
    """
    Compute the relative error between a pair of tensors x and y,
    which is defined as:

                            max_i |x_i - y_i]|
    rel_error(x, y) = -------------------------------
                      max_i |x_i| + max_i |y_i| + eps

    Inputs:
    - x, y: Tensors of the same shape
    - eps: Small positive constant for numeric stability

    Returns:
    - rel_error: Scalar giving the relative error between x and y
    """
    """ returns relative error between x and y """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot

################# Data #################

def get_toy_data(
    num_inputs=5,
    input_size=4,
    hidden_size=10,
    num_classes=3,
    dtype=torch.float32,
    device="cuda"
):
    """
    Get toy data for use when developing a two-layer-net.

    Inputs:
    - num_inputs: Integer N giving the data set size
    - input_size: Integer D giving the dimension of input data
    - hidden_size: Integer H giving the number of hidden units in the model
    - num_classes: Integer C giving the number of categories
    - dtype: torch datatype for all returned data
    - device: device on which the output tensors will reside

    Returns a tuple of:
    - toy_X: `dtype` tensor of shape (N, D) giving data points
    - toy_y: int64 tensor of shape (N,) giving labels, where each element is an
      integer in the range [0, C)
    - params: A dictionary of toy model parameters, with keys:
      - 'W1': `dtype` tensor of shape (D, H) giving first-layer weights
      - 'b1': `dtype` tensor of shape (H,) giving first-layer biases
      - 'W2': `dtype` tensor of shape (H, C) giving second-layer weights
      - 'b2': `dtype` tensor of shape (C,) giving second-layer biases
    """
    N = num_inputs
    D = input_size
    H = hidden_size
    C = num_classes

    # We set the random seed for repeatable experiments.
    reset_seed(0)

    # Generate some random parameters, storing them in a dict
    params = {}
    params["W1"] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
    params["b1"] = torch.zeros(H, device=device, dtype=dtype)
    params["W2"] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)
    params["b2"] = torch.zeros(C, device=device, dtype=dtype)

    # Generate some random inputs and labels
    toy_X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
    if N != 5:
        toy_y=torch.randint(C,(N,))
    else:   
        toy_y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

    return toy_X, toy_y, params

def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - x: `x_dtype` tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y

def cifar10(num_train=None, num_test=None, x_dtype=torch.float32):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: [Optional] How many samples to keep from the training set.
      If not provided, then keep the entire training set.
    - num_test: [Optional] How many samples to keep from the test set.
      If not provided, then keep the entire test set.
    - x_dtype: [Optional] Data type of the input image

    Returns:
    - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    
    download = not os.path.isdir("cifar-10-batches-py")
    dset_train = CIFAR10(root=".", download=download, train=True)
    dset_test = CIFAR10(root=".", train=False)
    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)
    return x_train, y_train, x_test, y_test

def preprocess_cifar10(
    show_examples=True,
    bias_trick=False,
    flatten=True,
    validation_ratio=0.2,
    dtype=torch.float32,
    device="cuda"
):
    """
    Returns a preprocessed version of the CIFAR10 dataset, automatically
    downloading if necessary. We perform the following steps:

    (0) [Optional] Visualize some images from the dataset
    (1) Normalize the data by subtracting the mean
    (2) Reshape each image of shape (3, 32, 32) into a vector of shape (3072,)
    (3) [Optional] Bias trick: add an extra dimension of ones to the data
    (4) Carve out a validation set from the training set

    Inputs:
    - device: If "cuda", move the entire dataset to the GPU
    - validation_ratio: Float in the range (0, 1) giving the fraction of the train
      set to reserve for validation
    - bias_trick: Boolean telling whether or not to apply the bias trick
    - show_examples: Boolean telling whether or not to visualize data samples
    - dtype: Optional, data type of the input image X

    Returns a dictionary with the following keys:
    - 'X_train': `dtype` tensor of shape (N_train, D) giving training images
    - 'X_val': `dtype` tensor of shape (N_val, D) giving val images
    - 'X_test': `dtype` tensor of shape (N_test, D) giving test images
    - 'y_train': int64 tensor of shape (N_train,) giving training labels
    - 'y_val': int64 tensor of shape (N_val,) giving val labels
    - 'y_test': int64 tensor of shape (N_test,) giving test labels

    N_train, N_val, and N_test are the number of examples in the train, val, and
    test sets respectively. The precise values of N_train and N_val are determined
    by the input parameter validation_ratio. D is the dimension of the image data;
    if bias_trick is False, then D = 32 * 32 * 3 = 3072;
    if bias_trick is True then D = 1 + 32 * 32 * 3 = 3073.
    """
    X_train, y_train, X_test, y_test = cifar10(x_dtype=dtype)

    # Move data to the GPU
    if device=="cuda":
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    elif device=="mps":
        mps_device = torch.device("mps")
        X_train = X_train.to(mps_device)
        y_train = y_train.to(mps_device)
        X_test = X_test.to(mps_device)
        y_test = y_test.to(mps_device)
    else:
        try:
            hw_device = torch.device(device)
            X_train = X_train.to(hw_device)
            y_train = y_train.to(hw_device)
            X_test = X_test.to(hw_device)
            y_test = y_test.to(hw_device)
        except RuntimeError as e:
            print("Invalid device "+str(device)+", leaving tensors on current device "+X_train.device.type)

    # 0. Visualize some examples from the dataset.
    if show_examples:
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        samples_per_class = 12
        samples = []
        reset_seed(0)
        for y, cls in enumerate(classes):
            plt.text(-4, 34 * y + 18, cls, ha="right")
            (idxs,) = (y_train == y).nonzero(as_tuple=True)
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                samples.append(X_train[idx])
        img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
        plt.imshow(tensor_to_image(img))
        plt.axis("off")
        plt.show()

    # 1. Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # 2. Reshape the image data into rows
    if flatten:
      X_train = X_train.reshape(X_train.shape[0], -1)
      X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Add bias dimension and transform into columns
    if bias_trick:
        ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
        X_train = torch.cat([X_train, ones_train], dim=1)
        ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
        X_test = torch.cat([X_test, ones_test], dim=1)

    # 4. take the validation set from the training set
    # Note: It should not be taken from the test set
    # For random permutation, you can use torch.randperm or torch.randint
    # But, for this homework, we use slicing instead.
    num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
    num_validation = X_train.shape[0] - num_training

    # return the dataset
    data_dict = {}
    data_dict["X_val"] = X_train[num_training : num_training + num_validation]
    data_dict["y_val"] = y_train[num_training : num_training + num_validation]
    data_dict["X_train"] = X_train[0:num_training]
    data_dict["y_train"] = y_train[0:num_training]

    data_dict["X_test"] = X_test
    data_dict["y_test"] = y_test
    return data_dict

def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None

    #take all samples if batch_size > num_train
    if batch_size >= num_train:
        batch_size=num_train
    ix=torch.randint(num_train, (batch_size,))
    X_batch=X[ix,:]
    y_batch=y[ix]

    return X_batch, y_batch

################# Visualizations #################

def plot_stats(stat_dict):
    # Plot the loss function and train / validation accuracies
    plt.subplot(1, 2, 1)
    plt.plot(stat_dict["loss_history"], "o")
    plt.title("Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(stat_dict["train_acc_history"], "o-", label="train")
    plt.plot(stat_dict["val_acc_history"], "o-", label="val")
    plt.title("Classification Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Accuracy")
    plt.legend()

    plt.gcf().set_size_inches(14, 4)
    plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    # print(Xs.shape)
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = torch.min(img), torch.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

# Visualize the weights of the network
def show_net_weights(net):
    W1 = net.params["W1"]
    W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
    plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
    plt.gca().axis("off")
    plt.show()

def plot_acc_curves(stat_dict):
    plt.subplot(1, 2, 1)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats["train_acc_history"], label=str(key))
    plt.title("Train Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Accuracy")

    plt.subplot(1, 2, 2)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats["val_acc_history"], label=str(key))
    plt.title("Validation Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Accuracy")
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()

def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names (e.g.) ['plane', 'car', 'bird', 'cat',
      'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    Outputs:
    - An grid-image that visualize samples_per_class number of samples per class
    """

    # Protected lazy import.
    from torchvision.utils import make_grid

    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(
            -4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha="right"
        )
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)