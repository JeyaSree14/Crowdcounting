# # HealthCare Imaging Analysis - anomaly possible or not

import cv2
import numpy as np

# Load the X-ray image
image = cv2.imread('Brain.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found.")
    exit()

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold to detect bright regions (anomalies)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours of these bright regions
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert grayscale to BGR to draw colored contours
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw contours
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:  # adjust sensitivity
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Possible Anomaly", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show the result
cv2.imshow("Detected Anomalies", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


