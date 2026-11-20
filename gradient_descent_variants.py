"""
TYBSc Artificial Neural Network Assignment
Topic: Comparison of Gradient Descent Variants
Roll no: 8055

1. AIM
To implement and compare three Gradient Descent algorithms, namely Batch Gradient Descent, Stochastic Gradient Descent and Mini-Batch Gradient Descent, by training a small Artificial Neural Network for a regression problem.
The comparison is based on convergence behaviour, final Mean Squared Error (MSE) and computation time.

2. Theory
Gradient Descent is an optimization algorithm used to minimize the error of a neural network by updating weights and biases.
- Batch GD: Uses the complete dataset for each update.
- Stochastic GD: Uses one sample for each update.
- Mini-Batch GD: Uses a small group of samples for each update.

3. Working Principle
1. Generate a dataset containing 240 samples.
2. Normalize input and output values.
3. Create a neural network with 1 input, 12 hidden and 1 output neuron.
4. Perform forward propagation to generate predictions.
5. Calculate error using MSE.
6. Perform backpropagation to calculate gradients.
7. Update weights using Gradient Descent.
8. Repeat training for 300 epochs.
9. Compare final loss and execution time.

4. Calculations
Mean Squared Error:
\[
MSE=\frac{1}{N}\sum(y_i-\hat y_i)^2
\]
Weight Update:
\[
W_{new}=W_{old}-\alpha \times Gradient
\]
Where α is the learning rate.
Training Parameters:
- Samples = 240
- Epochs = 300
- Mini-batch size = 24
Total Updates:
Method	Total updates
Batch GD	300
Stochastic GD	72,000
Mini-Batch GD	3,000

5. EXPLANATIONS
a. Batch Gradient Descent
- Uses all 240 training samples to calculate gradients.
- Performs one weight update per epoch.
- It achieved a final loss of 0.040490.
- Its execution time was 0.065281 seconds.
- It was the fastest method in this experiment but produced the highest final error.
b. Stochastic Gradient Descent
- Uses one sample at a time to calculate gradients.
- Performs 240 updates per epoch.
- It achieved the lowest final loss of 0.009940.
- Its execution time was 2.844225 seconds.
- It produced the best final error but required the longest execution time.
c. Mini-Batch Gradient Descent
- Uses 24 samples in each mini-batch.
- Performs 10 updates per epoch.
- It achieved a final loss of 0.011507.
- Its execution time was 0.625684 seconds.
- It provided a balance between training accuracy and computation time.

7. Result
All three Gradient Descent algorithms were successfully implemented. Stochastic GD achieved the lowest MSE of 0.009940, while Batch GD had the lowest execution time.

8. Conclusion
The experiment shows that different Gradient Descent methods have different advantages. Stochastic GD provides better final accuracy, Batch GD provides faster execution, and Mini-Batch GD provides a balance between accuracy and speed.
"""

import csv
import time
import numpy as np
import matplotlib.pyplot as plt



###import numpy as np
import matplotlib.pyplot as plt
import time
import csv
###
# 1. Create Dataset
def create_dataset(n=240):
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, (n, 1))
    y = np.sin(1.5*x) + 0.3*x + rng.normal(0, 0.08, (n, 1))
    return (x-x.mean())/x.std(), (y-y.mean())/y.std()

# 2. Initialize Neural Network
def initialize():
    rng = np.random.default_rng(10)
    return {
        "W1": rng.normal(0, 1, (1, 12)),
        "b1": np.zeros((1, 12)),
        "W2": rng.normal(0, np.sqrt(1/12), (12, 1)),
        "b2": np.zeros((1, 1))
    }

# 3. Forward Propagation
def forward(x, p):
    a = np.tanh(x @ p["W1"] + p["b1"])
    return a @ p["W2"] + p["b2"], a

# 4. Calculate Gradients
def backward(x, y, pred, a, p):
    m = len(x)
    dz2 = 2*(pred-y)/m
    da = dz2 @ p["W2"].T
    dz1 = da*(1-a**2)

    return {
        "W1": x.T @ dz1,
        "b1": dz1.sum(axis=0, keepdims=True),
        "W2": a.T @ dz2,
        "b2": dz2.sum(axis=0, keepdims=True)
    }

# 5. Train Model
def train(x, y, method, lr, epochs=300, bs=24):
    p = initialize()
    losses = []
    start = time.perf_counter()

    rng = np.random.default_rng(100)

    for _ in range(epochs):
        idx = rng.permutation(len(x))
        X, Y = x[idx], y[idx]

        size = len(x) if method == "Batch GD" else (
            1 if method == "Stochastic GD" else bs)

        for i in range(0, len(x), size):
            xb, yb = X[i:i+size], Y[i:i+size]
            pred, a = forward(xb, p)
            g = backward(xb, yb, pred, a, p)

            for k in p:
                p[k] -= lr*g[k]

        pred, _ = forward(x, p)
        losses.append(np.mean((pred-y)**2))

    return losses, time.perf_counter()-start


# 6. Main Program
x, y = create_dataset()
epochs, bs = 300, 24

methods = [
    ("Batch GD", 0.03),
    ("Stochastic GD", 0.003),
    ("Mini-Batch GD", 0.02)
]

results = {}

for method, lr in methods:
    loss, t = train(x, y, method, lr, epochs, bs)
    results[method] = {
        "losses": loss,
        "final_loss": loss[-1],
        "time": t,
        "learning_rate": lr
    }

# 7. Display Comparison Table
print("\nARTIFICIAL NEURAL NETWORK")
print("Comparison of Gradient Descent Variants")
print("-" * 75)

print(f"\nTraining samples : {len(x)}")
print(f"Epochs           : {epochs}")
print(f"Mini-batch size  : {bs}")

print("\nCOMPARISON RESULTS")
print("-" * 75)
print(f"{'Method':<20}{'Learning Rate':<18}{'Final Loss':<17}{'Time (sec)':<12}")
print("-" * 75)

for method, _ in methods:
    r = results[method]
    print(f"{method:<20}{r['learning_rate']:<18.4f}"
          f"{r['final_loss']:<17.6f}{r['time']:<12.6f}")

print("-" * 75)

# 8. Plot Graph
for method, r in results.items():
    plt.plot(range(1, epochs+1), r["losses"], label=method)

plt.title("Convergence Comparison of Gradient Descent Variants")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("gradient_descent_comparison.png")
plt.show()

# 9. Save CSV Results
with open("gradient_descent_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Learning Rate", "Final Loss", "Time (seconds)"])

    for method, r in results.items():
        writer.writerow([method, r["learning_rate"],
                         r["final_loss"], r["time"]])

print("\nFiles created successfully:")
print("1. gradient_descent_comparison.png")
print("2. gradient_descent_results.csv")
print("\nProgram completed.")
