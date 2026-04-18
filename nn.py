"""
nn.py — Python driver for nn_lib.c via ctypes

"""

import ctypes
import numpy as np
import os
import sys

#Load the shared library
if sys.platform == "win32":
    lib_path = os.path.join(os.path.dirname(__file__), "nn_lib.dll")
else:
    lib_path = os.path.join(os.path.dirname(__file__), "nn_lib.so")

lib = ctypes.CDLL(lib_path)
lib.init_network.argtypes = []
lib.init_network.restype  = None

lib.train.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # X
    ctypes.POINTER(ctypes.c_double),   # Y
    ctypes.c_int,                      # epochs
    ctypes.c_double,                   # lr
    ctypes.POINTER(ctypes.c_double),   # loss_out
]
lib.train.restype = None

lib.predict.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # X
    ctypes.POINTER(ctypes.c_double),   # out
]
lib.predict.restype = None

lib.destroy_network.argtypes = []
lib.destroy_network.restype  = None

#XOR data
X = np.ascontiguousarray([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float64)

Y = np.ascontiguousarray([0, 1, 1, 0], dtype=np.float64)

#Convert numpy array to ctypes pointer
def ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

#Train
EPOCHS = 20000
LR     = 1.0

# Allocate a numpy array to receive the loss curve from C
loss_history = np.zeros(EPOCHS, dtype=np.float64)

print("Initialising network...")
lib.init_network()

print(f"Training for {EPOCHS} epochs...")
lib.train(ptr(X), ptr(Y), ctypes.c_int(EPOCHS), ctypes.c_double(LR), ptr(loss_history))

# Print loss at intervals — loss_history is now filled by C
for i in range(0, EPOCHS, 2000):
    print(f"  Epoch {i:5d} | Loss: {loss_history[i]:.4f}")

#Predicting function:
predictions = np.zeros(4, dtype=np.float64)
lib.predict(ptr(X), ptr(predictions))

print("\nFinal predictions:")
labels = [[0,0],[0,1],[1,0],[1,1]]
targets = [0, 1, 1, 0]
for i, (inp, pred, tgt) in enumerate(zip(labels, predictions, targets)):
    print(f"  {inp} -> {pred:.3f}  (target: {tgt})")

#Destroy the network (Clean Up)
lib.destroy_network()
print("\nNetwork destroyed. Memory freed.")

#Optional: Plot the loss curve
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='#534AB7', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("XOR training loss (C backend)")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Loss curve saved to loss_curve.png")
except ImportError:
    print("(matplotlib not installed — skipping loss plot)")