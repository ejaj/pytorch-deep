import numpy as np
import matplotlib.pyplot as plt
from plots.ch_4 import plot_images


def sample_generate_image():
    # Image dimensions
    img_size = 5
    # Starting row for the upward diagonal
    start_row = 4  # This means we start from the bottom of the image

    # Create an empty (black) image
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Define the 'up' tuple as described
    up = (range(start_row, -1, -1), range(0, start_row + 1))
    print(up)

    # Drawing the upward diagonal
    # Note: Directly using 'up' for indexing in the way described isn't possible.
    # We need to convert the ranges into index positions for each pixel on the diagonal.
    for row, col in zip(up[0], up[1]):
        print(row, col)
        img[row, col] = 255  # Set the pixel to white

    plt.imshow(img, cmap='gray')
    plt.show()


def generate_image(start, target, fill=1, img_size=10):
    # Generates empty image
    img = np.zeros((img_size, img_size), dtype=float)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    if target == 0:
        if start_row is None:
            img[:, start_col] = fill
        else:
            img[start_row, :] = fill
    else:
        if start_col == 0:
            start_col = 1

        if target == 1:
            if start_row is not None:
                up = (range(start_row, -1, -1),
                      range(0, start_row + 1))
            else:
                up = (range(img_size - 1, start_col - 1, -1),
                      range(start_col, img_size))
            img[up] = fill
        else:
            if start_row is not None:
                down = (range(start_row, img_size, 1),
                        range(0, img_size - start_row))
            else:
                down = (range(0, img_size - 1 - start_col + 1),
                        range(start_col, img_size))
            img[down] = fill

    return 255 * img.reshape(1, img_size, img_size)


def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
    np.random.seed(seed)

    starts = np.random.randint(-(img_size - 1), img_size, size=(n_images,))
    targets = np.random.randint(0, 3, size=(n_images,))
    images = np.array([generate_image(s, t, img_size=img_size)
                       for s, t in zip(starts, targets)], dtype=np.uint8)

    if binary:
        targets = (targets > 0).astype(int)

    return images, targets


if __name__ == "__main__":
    images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=13)
    # print(images)
    # fig = plot_images(images, labels, n_plot=30)
    # sample_generate_image()
