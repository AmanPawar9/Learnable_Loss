# Learnable_Loss
## CNN Classification: Standard vs. Learnable Loss (PyTorch Implementation)

## Overview

This repository provides PyTorch implementations for training Convolutional Neural Network (CNN) classifiers on various image datasets. It contrasts two approaches:

1.  **Standard Training:** A conventional CNN trained using standard cross-entropy loss (or equivalent, like Binary Cross Entropy on one-hot labels).
2.  **Learnable Loss Training:** A CNN trained alongside a *separate, learnable neural network* that dynamically defines the loss function. This approach incorporates constraints (non-negativity, convexity, Lipschitz) to guide the learning of a meaningful loss landscape.

The code originates from TensorFlow/Keras implementations and has been converted to PyTorch, offering a way to explore these training paradigms within the PyTorch ecosystem. The primary focus of this repository is the PyTorch code.

## Features

* **Multiple Dataset Support:** Easily run experiments on MNIST, FashionMNIST, CIFAR-10, CIFAR-100, and Oxford-IIIT Pet datasets.
* **Standard CNN Implementation (`pytorch_classifier.py`):** A baseline CNN model trained with standard practices.
* **Learnable Loss CNN Implementation (`pytorch_learnable_loss.py`):** Implements the concept of a learnable loss function with constraints, requiring higher-order gradient calculations.
* **Multiple Runs & Averaging:** Automatically runs experiments multiple times (`NUM_RUNS`) and plots averaged performance metrics with standard deviations.
* **Detailed Plotting:** Generates informative plots for analysis:
    * Training & Validation Accuracy curves (mean ± std dev)
    * Training & Validation Loss curves (mean ± std dev for validation task loss)
    * Constraint Penalty curves (for the learnable loss method, plotted on log scale where appropriate)
    * Learned Loss Function Visualization (for the learnable loss method, showing Energy vs. Probability)
    * **Per-Class F1 Score Bar Chart** (for the learnable loss method, generated from the last run)
* **Comprehensive Metrics:** Reports standard accuracy and provides detailed classification reports (Precision, Recall, F1-score) using `scikit-learn`.
* **Device Agnostic:** Runs on either CUDA GPU (if available) or CPU.

## File Structure
```
├── pytorch_classifier.py             # Script for standard CNN training
├── pytorch_learnable_loss.py         # Script for learnable loss CNN training
├── plots_pytorch/                    # Output directory for standard CNN plots
├── plots_learnable_loss_detailed_pytorch/ # Output directory for learnable loss plots
├── data/                             # Default directory for downloaded datasets (auto-created)
└── README.md                         # This file
```
## Requirements

The code is written in Python 3. You'll need the following libraries:

* PyTorch (`torch`)
* Torchvision (`torchvision`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Scikit-learn (`scikit-learn`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:** (It's recommended to use a virtual environment)
    ```bash
    pip install torch torchvision numpy matplotlib scikit-learn
    # Or install with specific CUDA version if needed, see: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    ```

## Usage

You can run each training script independently from the command line.

1.  **Run the Standard CNN Classifier:**
    ```bash
    python pytorch_classifier.py
    ```

2.  **Run the Learnable Loss CNN Classifier:**
    ```bash
    python pytorch_learnable_loss.py
    ```

**Notes:**

* The scripts will automatically download the required datasets to the `./data/` directory on their first run for each dataset.
* Training progress, epoch timings, and final averaged metrics will be printed to the console.
* Plots will be saved to the respective directories (`plots_pytorch/` and `plots_learnable_loss_detailed_pytorch/`).

## Configuration

You can modify hyperparameters and settings directly within the Python scripts (`.py` files):

* `LEARNING_RATE_CLASSIFIER`, `LEARNING_RATE_LOSS_NETWORK`: Learning rates for the respective optimizers.
* `BATCH_SIZE`: Number of samples per training batch.
* `EPOCHS`: Number of training epochs per run.
* `NUM_RUNS`: How many times to repeat the training for averaging results.
* `LAMBDA_CONST`: Weighting factor for the constraint losses in the learnable loss method.
* `PLOT_SAVE_DIR`: Change the output directory name for plots.
* `DATASETS` dictionary: You can comment out datasets if you only want to run on a subset. Note that the input shapes and number of classes are defined here.

## Output

* **Console Output:** Displays progress per epoch (losses, accuracy, timing), final averaged metrics over `NUM_RUNS`, and a detailed `sklearn` classification report for the last run of each dataset.
* **Plots:** Saved as PNG files in the corresponding plot directories.
    * **`pytorch_classifier.py` output (`plots_pytorch/`):**
        * `results_{dataset}_accuracy_plot_pytorch.png`: Mean Training/Validation Accuracy ± Std Dev.
        * `results_{dataset}_loss_log_plot_pytorch.png`: Training/Validation Loss from the last run.
    * **`pytorch_learnable_loss.py` output (`plots_learnable_loss_detailed_pytorch/`):**
        * `results_{dataset}_learnable_loss_detailed_plots_pytorch.png`: A 4-panel plot showing:
            * Avg. Training Total Loss vs Avg. Validation Task Loss ± Std Dev.
            * Avg. Validation Accuracy ± Std Dev.
            * Avg. Constraint Penalties (Non-negativity, Convexity, Lipschitz) over epochs.
            * Visualization of the learned loss energy E(p,y) vs. the probability assigned to the true class (p[0]) from the last run.
        * `f1_scores_{dataset}_pytorch.png`: A bar chart showing the per-class F1-scores calculated from the test set classification report of the *last run*.

## Learnable Loss Details

The `pytorch_learnable_loss.py` script implements a technique where the loss function itself is parameterized by a neural network (`LossNetwork`). Instead of using a fixed function like Cross-Entropy, this network takes the classifier's output probabilities (`p`) and the true one-hot labels (`y`) and outputs a scalar "energy" `E(p, y)`. The goal is for the network to *learn* a loss function suitable for the task.

To ensure the learned energy function behaves reasonably (e.g., minimal when `p` matches `y`), several constraints are added to the total loss during training, weighted by `LAMBDA_CONST`:

* **`L_task`**: The standard task loss (Binary Cross Entropy between `p` and `y_one_hot` in this implementation) to ensure the classifier still learns to predict correctly.
* **`L_nonneg`**: Penalizes negative energy outputs (`E`), encouraging the learned loss to be non-negative. ( `mean(relu(-E)^2)` )
* **`L_convex`**: Penalizes non-convexity of the energy function with respect to the predicted probabilities `p`. This is approximated by penalizing negative traces of the Hessian matrix (`d^2E / dp^2`). ( `mean(relu(-trace(Hessian))^2)` )
* **`L_lips`**: Encourages the gradient of the energy function with respect to `p` (`dE/dp`) to have a norm close to 1, enforcing a 1-Lipschitz constraint. ( `mean((norm(grad_E) - 1)^2)` )

The total loss optimized is `L_total = L_task + LAMBDA_CONST * (L_nonneg + L_convex + L_lips)`. Both the classifier and the loss network are updated based on the gradients of this total loss. This requires computing second-order derivatives (for the Hessian trace involved in `L_convex`).
