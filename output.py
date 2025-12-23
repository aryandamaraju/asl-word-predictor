from PIL import Image
import numpy as np

def read_ppm_p6(filename):
    with open(filename, "rb") as f:
        header = f.readline().strip()
        if header != b"P6":
            raise ValueError("Invalid PPM format (not P6)")

        # Read width, height, and max color value
        width, height = map(int, f.readline().strip().split())
        max_color = int(f.readline().strip())

        if max_color != 255:
            raise ValueError("Only 255 max color PPM files are supported")

        # Read pixel data
        pixel_data = np.frombuffer(f.read(), dtype=np.uint8)
        img = pixel_data.reshape((height, width, 3))  # Reshape to (H, W, 3)
    
    return Image.fromarray(img, "RGB")

# Load and display the PPM image
image = read_ppm_p6("output.ppm")
image.show()  # Opens the image
image.save("output.png")  # Save as PNG for easier access
