import cv2

# Load the image
image_path = 'data/1.jpg'  # Make sure to use the correct path to your image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR loads the image in the BGR color format

flipped_image = cv2.flip(image, 0)

# Display the original image
cv2.imshow('Original Image', image)

# Display the flipped image
cv2.imshow('Flipped Image', flipped_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()