import matplotlib.pyplot as plt

# experiment data
models = [
    "1 Conv 3x3",
    "1 Conv 4x4",
    "1 Conv 2x2",
    "1 Conv 3x3 (6 epochs)",
    "2 Conv layers"
]

accuracy = [
    0.980,
    0.981,
    0.979,
    0.980,
    0.985
]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)
plt.ylabel("Test Accuracy")
plt.ylim(0.95, 1.0)
plt.title("MNIST CNN Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()

