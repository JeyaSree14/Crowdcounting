import cv2
import numpy as np

# Video file or camera stream
cap = cv2.VideoCapture('crowd_video.mp4')  # Use 0 for webcam

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours (people-like blobs)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small movements
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count += 1

    # Estimate density as a heatmap
    heatmap = cv2.applyColorMap(fgmask, cv2.COLORMAP_JET)

    # Overlay density
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    # Show crowd count
    cv2.putText(overlay, f'Count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Crowd Counting and Density Estimation', overlay)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()