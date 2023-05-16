import cv2

# Load the image
def dd(p):
    image = cv2.imread(p, cv2.IMREAD_COLOR)

# Split the image into color channels
    b, g, r = cv2.split(image)

# Apply median filter to each channel
    b_denoised = cv2.medianBlur(b, 5)  # Adjust the kernel size (5) as needed
    g_denoised = cv2.medianBlur(g, 5)
    r_denoised = cv2.medianBlur(r, 5)

# Merge the denoised channels back into a color image
    denoised_image = cv2.merge((b_denoised, g_denoised, r_denoised))
    return denoised_image
# # Display the original and denoised images
# cv2.imshow('Original Image', gray_image)
# cv2.imshow('Denoised Image', denoised_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
