import numpy as np
import cv2
import imutils

def perform_processing(image: np.ndarray, character_dict: dict) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    # Ograniczenie obszaru, gdzie znajduje się tablica
    img = image
    img = cv2.resize(img, (800,600))
    img_og = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    cont, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_cont = img_og.copy()
    cv2.drawContours(img_cont, cont, -1, (0,255,0), 3)
    # cv2.imshow("test", img_cont)
    max_contour = max(cont, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    plate = img_og[y:y+h, x:x+w]
    plate = cv2.resize(plate, (900, 600))
    # cv2.imshow("test2", plate)
    # cv2.waitKey(0)

    #reiteracja
    img = plate
    img = cv2.resize(img, (800,600))
    img_og = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cont, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_cont = img_og.copy()
    cv2.drawContours(img_cont, cont, -1, (0,255,0), 3)
    cv2.imshow("test", img_cont)
    max_contour = max(cont, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    plate = img_og[y:y+h, x:x+w]
    plate = cv2.resize(plate, (900, 600))
    cv2.imshow("test2", plate)
    cv2.waitKey(0)

    return 'PO12345'