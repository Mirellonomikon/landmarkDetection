import cv2
import dlib
import numpy as np


def landmark(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    new_image = cv2.imread(image)
    # Convert the image color to grayscale
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
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
        t = 2
        # Display the landmarks
        width = new_image.shape[1]
        if width > 2000:
            t = 5
        #     new_image = imutils.resize(new_image, width=720)
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            if i < 16:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (0, 0, 255), t)
            elif 16 < i < 21:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 21 < i < 26:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 26 < i < 30:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 30 < i < 35:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 35 < i < 41:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 41 < i < 47:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)
            elif 47 < i < 67:
                (x2, y2) = shape[i + 1]
                cv2.line(new_image, (x, y), (x2, y2), (150, 200, 0), t)

        cv2.line(new_image, shape[48], shape[60], (150, 200, 0), t)
        cv2.line(new_image, shape[67], shape[60], (150, 200, 0), t)
        cv2.line(new_image, shape[64], shape[54], (150, 200, 0), t)
        cv2.line(new_image, shape[30], shape[33], (150, 200, 0), t)
        cv2.line(new_image, shape[36], shape[41], (150, 200, 0), t)
        cv2.line(new_image, shape[42], shape[47], (150, 200, 0), t)

    # Display the image

    return new_image
