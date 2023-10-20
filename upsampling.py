import cv2
import numpy as np

def bilinear_interpolation(image, scale_factor):
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Calculate the new dimensions based on the scale factor
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Create an empty image with the new dimensions
    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    # Compute the scale factors for mapping the new image coordinates to the original image coordinates
    scale_x = width / new_width
    scale_y = height / new_height

    # Perform bilinear interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding coordinates in the original image
            x_original = x * scale_x
            y_original = y * scale_y

            # Find the surrounding pixels in the original image
            x1 = int(x_original)
            y1 = int(y_original)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            # Calculate the fractional distances
            dx = x_original - x1
            dy = y_original - y1

            # Perform bilinear interpolation
            intensity = (1 - dx) * (1 - dy) * image[y1, x1] + \
                        dx * (1 - dy) * image[y1, x2] + \
                        (1 - dx) * dy * image[y2, x1] + \
                        dx * dy * image[y2, x2]

            # Assign the interpolated intensity to the corresponding pixel in the new image
            new_image[y, x] = intensity

    return new_image

# Load the medical image
image = cv2.imread('dataset/Testing/pituitary_tumor/image(15).jpg', cv2.IMREAD_GRAYSCALE)

# Specify the scale factor for upsampling
scale_factor = 2

# Perform bilinear interpolation
upsampled_image = bilinear_interpolation(image, scale_factor)

# Display the original and upsampled images
cv2.imshow('Original Image', image)
cv2.imshow('Upsampled Image', upsampled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
