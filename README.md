# MNIST Digit Classification using CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.  
Multiple CNN configurations were tested to analyze the effects of kernel size, model depth, pooling, and training epochs on model performance.

---

## Requirements

• Python 3.9+ (Python 3.11 recommended)  
• TensorFlow ≥ 2.15  
• NumPy

---

## Data Preprocessing

• Pixel values were normalized from `[0–255]` to `[0–1]`.  
• No resizing or augmentation was applied since MNIST digits are already standardized and centered.

---

## Model Architectures

### Shallow CNN

Input (28×28)
→ Reshape (28×28×1)
→ Conv2D
→ Flatten
→ Dense(10, Softmax)


### Deeper CNN

Input (28×28)
→ Reshape
→ Conv2D(16, 3×3)
→ Conv2D(32, 3×3)
→ Flatten
→ Dense(10, Softmax)


---

## Experimental Results

| Model Variant | Network Architecture | Epochs | Training Time | Test Accuracy |
|----------------|------------------------|--------|----------------|----------------|
| Single Conv (3×3) | Conv2D(16, 3×3) → Flatten | 3 | ~10 s | 0.980 |
| Single Conv (4×4) | Conv2D(16, 4×4) → Flatten | 3 | ~11 s | 0.981 |
| Single Conv (2×2) | Conv2D(16, 2×2) → Flatten | 3 | ~9 s | 0.979 |
| Single Conv (3×3) – Extended Training | Conv2D(16, 3×3) → Flatten | 6 | ~18 s | 0.980 |
| **Two Convolution Layers (Best)** | Conv2D(16, 3×3) → Conv2D(32, 3×3) → Flatten | 6 | ~22 s | **0.985** |

---

## Best Model Selection

The stacked two‐layer CNN using 3×3 kernels achieved the highest accuracy (98.5%).  
Increasing network depth proved more effective than increasing kernel size or training epochs alone.

---

## Key Observations

• Larger kernels slightly improved performance in shallow networks by increasing the receptive field.  
• Smaller kernels (2×2) underperformed due to limited spatial context.  
• Pooling layers were avoided to preserve fine spatial details necessary for digit classification.  
• Increasing epochs alone produced diminishing returns once convergence was reached.

---

## Limitations

• MNIST is a small and highly structured dataset — results may not generalize to complex image tasks.  
• No regularization or augmentation was used.  
• Training was performed on CPU only.

---

## How To Run

1. Clone the repository:

```bash
git clone https://github.com/sanjitaganesh/mnist-digit-classification-cnn
cd mnist-digit-classification-cnn

