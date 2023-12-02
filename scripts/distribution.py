import matplotlib.pyplot as plt
import numpy as np


def plot_circle(x, y, m):
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = x + m * np.cos(theta)
    circle_y = y + m * np.sin(theta)

    plt.plot(circle_x, circle_y, label="Circle")
    plt.scatter(x, y, color="red", label="Center Point")  # Mark the center point
    plt.title(f"Circle with Radius {m} at ({x}, {y})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Ensure equal scaling on both axes
    plt.show()


# Example usage
x_center = 2
y_center = 3
radius = 4
plot_circle(x_center, y_center, radius)
