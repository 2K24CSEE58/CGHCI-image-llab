import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, channels = image.shape

    pixel_count = width * height
    estimated_bytes = pixel_count * channels

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": list(image.shape),
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": "BGR",
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    blue, green, red = cv2.split(image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    new_width = width // 2
    new_height = height // 2

    downsampled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    output_path = os.path.join(output_dir, "pixel_views.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(red, cmap="gray")
    axes[0, 1].set_title("Red Channel")

    axes[0, 2].imshow(green, cmap="gray")
    axes[0, 2].set_title("Green Channel")

    axes[1, 0].imshow(blue, cmap="gray")
    axes[1, 0].set_title("Blue Channel")

    axes[1, 1].imshow(grayscale, cmap="gray")
    axes[1, 1].set_title("Grayscale")

    axes[1, 2].imshow(cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(
        f"Downsampled ({new_width} x {new_height})"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "original_size": [width, height],
        "downsampled_size": [new_width, new_height],
        "output_path": output_path,
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""

    os.makedirs(output_dir, exist_ok=True)

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brighter = np.clip(
        grayscale.astype(np.int16) + brightness_delta,
        0,
        255
    ).astype(np.uint8)

    higher_contrast = np.clip(
        grayscale.astype(np.float32) * contrast_factor,
        0,
        255
    ).astype(np.uint8)

    _, thresholded = cv2.threshold(
        grayscale,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    output_path = os.path.join(output_dir, "adjustments.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(grayscale, cmap="gray")
    axes[0, 0].set_title("Original Grayscale")

    axes[0, 1].imshow(brighter, cmap="gray")
    axes[0, 1].set_title(f"Brighter (+{brightness_delta})")

    axes[1, 0].imshow(higher_contrast, cmap="gray")
    axes[1, 0].set_title(f"Higher Contrast (x{contrast_factor})")

    axes[1, 1].imshow(thresholded, cmap="gray")
    axes[1, 1].set_title(f"Threshold ({threshold})")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path,
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""

    os.makedirs(output_dir, exist_ok=True)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.blur(
        grayscale,
        (kernel_size, kernel_size)
    )

    sobel_x_original = cv2.Sobel(
        grayscale,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y_original = cv2.Sobel(
        grayscale,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    edges_original = cv2.magnitude(
        sobel_x_original,
        sobel_y_original
    )

    sobel_x_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y_blurred = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    edges_blurred = cv2.magnitude(
        sobel_x_blurred,
        sobel_y_blurred
    )

    output_path = os.path.join(
        output_dir,
        "blur_and_edges.png"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(grayscale, cmap="gray")
    axes[0, 0].set_title("Grayscale")

    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title(f"Mean Blur ({kernel_size}x{kernel_size})")

    axes[1, 0].imshow(edges_original, cmap="gray")
    axes[1, 0].set_title("Sobel Edges - Original")

    axes[1, 1].imshow(edges_blurred, cmap="gray")
    axes[1, 1].set_title("Sobel Edges - Blurred")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "kernel_size": kernel_size,
        "output_path": output_path,
    }


def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1-4 and return their results together."""

    os.makedirs(output_dir, exist_ok=True)

    task1 = inspect_image(image_path)
    task2 = create_pixel_views(image_path, output_dir)
    task3 = create_adjustments(image_path, output_dir)
    task4 = create_blur_and_edges(image_path, output_dir)

    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
    }


def main() -> None:
    """Run the lab using the required repository paths."""

    image_path = "images/original.jpg"
    output_dir = "outputs"

    results = run_lab(image_path, output_dir)

    print("Task 1 - Image Information")
    print(results["task1"])

    print("\nTask 2 - Pixel Views")
    print(results["task2"])

    print("\nTask 3 - Adjustments")
    print(results["task3"])

    print("\nTask 4 - Blur and Edges")
    print(results["task4"])

    print("\nLab completed successfully.")


if __name__ == "__main__":
    main()