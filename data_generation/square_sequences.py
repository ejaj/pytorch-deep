import numpy as np


def generate_sequences(n=128, variable_len=False, seed=13):
    basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    np.random.seed(seed)
    bases = np.random.randint(4, size=n)
    if variable_len:
        lengths = np.random.randint(3, size=n) + 2
    else:
        lengths = [4] * n
    directions = np.random.randint(2, size=n)
    points = [
        basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d * 2 - 1)][:l] + np.random.randn(l, 2) * 0.1
        for b, d, l in zip(bases, directions, lengths)]
    return points, directions


if __name__ == '__main__':
    # Generate sequences with fixed length
    points_fixed, directions_fixed = generate_sequences(n=5, variable_len=False, seed=42)

    # Generate sequences with variable length
    points_variable, directions_variable = generate_sequences(n=5, variable_len=True, seed=42)

    print(points_fixed[0])
    print(points_variable[0])
