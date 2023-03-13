import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to create a mask
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a new image for the output
output = np.zeros(img.shape, dtype=np.uint8)

# Loop over the contours
for i, contour in enumerate(contours):
    # Calculate the contour area
    area = cv2.contourArea(contour)

    # If the area is too small, skip this contour
    if area < 10*10:
        continue

    # Draw the contour on the output image
    cv2.drawContours(output, contours, i, (255, 255, 255), -1)

# Apply the mask to the input image
masked_img = cv2.bitwise_and(img, output)

# Create a new alpha channel with values based on the mask
alpha = np.zeros(mask.shape[:2], dtype=np.uint8)
alpha[output[:,:,0] > 0] = 255

# Add the alpha channel to the masked image
bgra = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
bgra[:, :, 3] = alpha

# Save the masked image to a PNG file with transparency
cv2.imwrite('image_masked.png', bgra)

# Display the original and masked images
cv2.imshow('Original Image', img)
cv2.imshow('Masked Image', masked_img)
cv2.waitKey(0)
