# Educational Neural Network from Scratch

A transparent, framework-free implementation of a Multi-Layer Feedforward Neural Network written in Python. Built from scratch using NumPy to expose and trace the foundational matrix mathematics and calculus driving deep learning.

---

## Origin & Inspiration

In January 2022, months before the release of ChatGPT and the subsequent rise of conversational AI, I built this program. Its creation was inspired by a machine learning course I was taking at the time, combined with a personal curiosity to understand exactly what occurs behind the scenes of a model. 

Today, artificial intelligence is widely accessible, but the programming behind it has become highly abstract. Modern developers often call a single pre-built function without needing to interact with the underlying training mechanics or weight matrix routing. This repository presents the core, foundational structure of a neural network—untouched by modern wrappers—exactly as I parsed it from mathematical principles in 2022.

---

## Project Scope & Private Disclaimer

This repository is maintained by a private individual strictly for educational and pedagogical purposes. 

* **Intent:** A clear mathematical sandbox. It is designed to allow a user to open a standard debugger and trace every forward matrix multiplication, bias augmentation, and backpropagation gradient calculation step-by-step.
* **Limitations:** This code is not optimized for production scale. It does not utilize GPU acceleration (CUDA) or complex parallel processing architectures, prioritizing readability and mathematical traceability over performance.

---

## Repository Architecture

The codebase is split into four cohesive components:

1. **`THEORY.md`:** An introductory, lecture-style guide detailing the structural topology, matrix dimensions, and forward propagation equations used to design the network.
2. **`network.py`:** Contains the core `NeuralNetwork` class. It manages arbitrary layer topology initialization, executes vectorized forward propagation passes, computes cross-entropy loss, and derives analytical gradients via backpropagation.
3. **`network_handler.py`:** A utility layer that handles data administration separate from the mathematics. It manages data shuffling, mini-batch slicing, dataset partitioning (train/validation/test), one-hot vector translation, and model state serialization (`.pkl` checkpointing).
4. **`example1.py`:** A fully commented walkthrough script that establishes a synthetic classification problem, runs the optimization loops using `scipy.optimize.fmin_cg`, charts the training history, and prints out real-time prediction audits.

---

## Core Mathematics

To maintain complete algorithmic transparency, all operations are written natively using linear algebra layouts. For any given layer $l$, the pre-activation matrix $Z$ and final activation states $A$ are mapped out explicitly:

$$Z^{l} = \tilde{A}^{l-1} \cdot \Theta^{l}$$
$$A^{l} = g(Z^{l})$$

Where $\tilde{A}^{l-1}$ is the activation matrix of the preceding layer prepended with a static column of ones to account for the Bias Neurons, $\Theta^{l}$ represents the parameter weight matrix, and $g(z)$ represents the element-wise Sigmoid Activation Function:

$$g(z) = \frac{1}{1 + e^{-z}}$$

For the complete breakdown of indexing, dimensionality matching, and batch processing matrices, refer directly to the [THEORY.md](THEORY.md) file.

---

## Getting Started

### Prerequisites
The project avoids external heavy deep learning dependencies and requires only standard scientific Python packages:
* Python 3.x
* NumPy
* SciPy
* Matplotlib

### Running the Example
To see the system initialize, train, map out performance logs, and output out-of-sample prediction metrics, execute the sandbox file from your terminal:

```bash
python example1.py
