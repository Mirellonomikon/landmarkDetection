import cv2
import dlib
import imutils
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread("test3.jpg")

# Convert the image color to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect the face
rects = detector(gray, 1)

for rect in rects:
    # Get the landmark points
    shape = predictor(gray, rect)
    # Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    # Display the landmarks
    for i, (x, y) in enumerate(shape):
        # Draw the circle to mark the keypoint
        if i < 16:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (0, 0, 255), 4)
        elif 16 < i < 21:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 4)
        elif 21 < i < 26:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 4)
        elif 26 < i < 30:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (255, 0, 0), 4)
        elif 30 < i < 35:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (255, 0, 0), 4)
        elif 35 < i < 41:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 4)
        elif 41 < i < 47:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 4)
        elif 47 < i < 67:
            (x2, y2) = shape[i + 1]
            cv2.line(image, (x, y), (x2, y2), (150, 200, 0), 4)

    cv2.line(image, shape[48], shape[60], (150, 200, 0), 4)
    cv2.line(image, shape[67], shape[60], (150, 200, 0), 4)
    cv2.line(image, shape[64], shape[54], (150, 200, 0), 4)
    cv2.line(image, shape[30], shape[33], (255, 0, 0), 4)
    cv2.line(image, shape[36], shape[41], (0, 255, 0), 4)
    cv2.line(image, shape[42], shape[47], (0, 255, 0), 4)

# Display the image
width = image.shape[1]
if width > 720:
    image = imutils.resize(image, width=720)

cv2.imshow('Face Landmarks Detection', image)
cv2.waitKey(0)
