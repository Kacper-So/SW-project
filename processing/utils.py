import numpy as np
import cv2
import imutils

def perform_processing(image: np.ndarray, character_dict: dict) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    img = image
    img = cv2.resize(img, (800,600))
    img_og = img.copy()
    img = cv2.GaussianBlur(img, (7, 7), 0)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 200])
    img = cv2.inRange(img, lower_blue, upper_blue)
    img = cv2.Canny(img, 100, 200)
    img = cv2.dilate(img, kernel, iterations=1)
    cont, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_cont = img_og.copy()
    max_contour = max(cont, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    zoom = img_og[y - int(h * 0.75):y + int(h * 1.75) , int(0.9 * x):x + (w * 13)]
    zoom = cv2.resize(zoom, (900, 600))

    zoom_og = zoom.copy()
    zoom = cv2.GaussianBlur(zoom, (9, 9), 0)
    zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    zoom = cv2.adaptiveThreshold(zoom, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    zoom = cv2.dilate(zoom, kernel, iterations=1)
    zoom = cv2.erode(zoom, kernel, iterations=1)
    zoom = cv2.Canny(zoom, 100, 200)
    zoom = cv2.dilate(zoom, kernel, iterations=2)
    zoom = cv2.erode(zoom, kernel, iterations=1)
    cont, hierarchy = cv2.findContours(zoom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    zoom_cont = zoom_og.copy()
    cv2.drawContours(zoom_cont, cont, -1, (0,255,0), 3)

    max_contour = max(cont, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    plate = zoom_og[y:y+h, x:x+w]
    plate = cv2.resize(plate, (900, 600))

    part = plate
    part = cv2.resize(part, (900,600))
    part_og = part.copy()
    part = cv2.GaussianBlur(part, (3, 3), 0)
    kernel = np.ones((2, 2), np.uint8)
    part = cv2.dilate(part, kernel, iterations=1)
    part = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 90])
    upper_blue = np.array([130, 255, 255])
    part = cv2.inRange(part, lower_blue, upper_blue)
    cont, hierarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    part_cont = part_og.copy()
    cv2.drawContours(part_cont, cont, -1, (0,255,0), 3)
    max_contour = max(cont, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    src = np.array([[0, 0], [899, 599 - (1.1 * h)], [899, 599], [0, (1.1 * h)]], dtype=np.float32)
    dst = np.array([[0, 0], [900 - 1, 0], [900 - 1, 600 - 1], [0, 600 -1]], dtype=np.float32)
    Matrix = cv2.getPerspectiveTransform(src, dst)
    plate_straight = part_og.copy()
    plate_upper = cv2.warpPerspective(part_og, Matrix, (900, 600))
    src = np.array([[0, 599 - (1.1 * h)], [899, 0], [899, (1.1 * h)], [0, 599]], dtype=np.float32)
    dst = np.array([[0, 0], [900 - 1, 0], [900 - 1, 600 - 1], [0, 600 -1]], dtype=np.float32)
    Matrix = cv2.getPerspectiveTransform(src, dst)
    plate_lower = cv2.warpPerspective(part_og, Matrix, (900, 600))
    cv2.imshow("first_variant", plate_straight)
    cv2.imshow('second_variant', plate_upper)
    cv2.imshow('third_variant', plate_lower)
    cv2.waitKey(0)
    return 'PO12345'