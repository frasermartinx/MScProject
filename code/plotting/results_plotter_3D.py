import matplotlib.pyplot as plt
import numpy as np


def plotter_3D(input,prediction,solution,X,Y):

    R,Theta = np.meshgrid(X, Y)

    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X, Y, input, cmap='viridis')
    ax.set_title("Input")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X, Y, prediction, cmap='viridis')
    ax.set_title("Prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(X, Y, solution, cmap='viridis')
    ax.set_title("Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    plt.show()

    