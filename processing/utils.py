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
    approx = cv2.approxPolyDP(max_contour, 0.015 * cv2.arcLength(max_contour, True), True)
    print(approx)
    # x, y, w, h = cv2.boundingRect(max_contour)
    # zoom = img_og[y - int(h * 0.75):y + int(h * 1.75) , int(0.9 * x):x + (w * 14)]
    # zoom = img_og[y:y + h, x:x + w]
    # zoom = cv2.resize(zoom, (900, 600))

    # zoom_og = zoom.copy()
    # zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    # zoom = cv2.GaussianBlur(zoom, (3, 3), 0)
    # zoom = cv2.adaptiveThreshold(zoom, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)
    # kernel = np.ones((3, 3), np.uint8)
    # zoom = cv2.morphologyEx(zoom, cv2.MORPH_OPEN, kernel)
    # # zoom = cv2.Canny(zoom, 100, 200)
    # cont, hierarchy = cv2.findContours(zoom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # zoom_cont = zoom_og.copy()
    # cv2.drawContours(zoom_cont, cont, -1, (0,255,0), 1)
    # cv2.imshow("xd", zoom)
    # cv2.imshow("zoom_cont", zoom_cont)
    # cv2.waitKey(0)
    # max_contour = max(cont, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(max_contour)
    # plate = zoom_og[y:y+h, x:x+w]
    # plate = cv2.resize(plate, (900, 600))

    # part = plate
    # part = cv2.resize(part, (900,600))
    # part_og = part.copy()
    # part = cv2.GaussianBlur(part, (3, 3), 0)
    # kernel = np.ones((2, 2), np.uint8)
    # part = cv2.dilate(part, kernel, iterations=1)
    # part = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([100, 150, 90])
    # upper_blue = np.array([130, 255, 255])
    # part = cv2.inRange(part, lower_blue, upper_blue)
    # cont, hierarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # part_cont = part_og.copy()
    # cv2.drawContours(part_cont, cont, -1, (0,255,0), 3)
    # max_contour = max(cont, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(max_contour)
    # src = np.array([[0, 0], [899, 599 - (1.1 * h)], [899, 599], [0, (1.1 * h)]], dtype=np.float32)
    # dst = np.array([[0, 0], [900 - 1, 0], [900 - 1, 600 - 1], [0, 600 -1]], dtype=np.float32)
    # Matrix = cv2.getPerspectiveTransform(src, dst)
    # plate_straight = part_og.copy()
    # plate_upper = cv2.warpPerspective(part_og, Matrix, (900, 600))
    # src = np.array([[0, 599 - (1.1 * h)], [899, 0], [899, (1.1 * h)], [0, 599]], dtype=np.float32)
    # dst = np.array([[0, 0], [900 - 1, 0], [900 - 1, 600 - 1], [0, 600 -1]], dtype=np.float32)
    # Matrix = cv2.getPerspectiveTransform(src, dst)
    # plate_lower = cv2.warpPerspective(part_og, Matrix, (900, 600))

    # # first_variant = plate_straight.copy()
    # # first_variant = cv2.GaussianBlur(first_variant, (9, 9), 0)
    # # first_variant = cv2.dilate(first_variant, kernel, iterations=1)
    # # first_variant = cv2.cvtColor(first_variant, cv2.COLOR_BGR2HSV)
    # # lower_black = np.array([0, 0, 0])
    # # upper_black = np.array([180, 255, 90])
    # # first_variant = cv2.inRange(first_variant, lower_black, upper_black)
    # # kernel = np.ones((2, 2), np.uint8)
    # # first_variant = cv2.dilate(first_variant, kernel, iterations=1)
    # # first_variant = cv2.erode(first_variant, kernel, iterations=1)
    # # cont, hierarchy = cv2.findContours(first_variant, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # first_variant_cont = plate_straight.copy()
    # # cv2.drawContours(first_variant_cont, cont, -1, (0,255,0), 3)
    # # cv2.imshow("first_variant", first_variant_cont)

    # # second_variant = plate_upper.copy()
    # # second_variant = cv2.GaussianBlur(second_variant, (9, 9), 0)
    # # second_variant = cv2.dilate(second_variant, kernel, iterations=1)
    # # second_variant = cv2.cvtColor(second_variant, cv2.COLOR_BGR2HSV)
    # # lower_black = np.array([0, 0, 0])
    # # upper_black = np.array([180, 255, 90])
    # # second_variant = cv2.inRange(second_variant, lower_black, upper_black)
    # # kernel = np.ones((2, 2), np.uint8)
    # # second_variant = cv2.dilate(second_variant, kernel, iterations=1)
    # # second_variant = cv2.erode(second_variant, kernel, iterations=1)
    # # cont, hierarchy = cv2.findContours(second_variant, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # second_variant_cont = plate_upper.copy()
    # # cv2.drawContours(second_variant_cont, cont, -1, (0,255,0), 3)
    # # cv2.imshow("second_variant", second_variant_cont)

    # # third_variant = plate_lower.copy()
    # # third_variant = cv2.GaussianBlur(third_variant, (9, 9), 0)
    # # third_variant = cv2.dilate(third_variant, kernel, iterations=1)
    # # third_variant = cv2.cvtColor(third_variant, cv2.COLOR_BGR2HSV)
    # # lower_black = np.array([0, 0, 0])
    # # upper_black = np.array([180, 255, 90])
    # # third_variant = cv2.inRange(third_variant, lower_black, upper_black)
    # # kernel = np.ones((2, 2), np.uint8)
    # # third_variant = cv2.dilate(third_variant, kernel, iterations=1)
    # # third_variant = cv2.erode(third_variant, kernel, iterations=1)
    # # cont, hierarchy = cv2.findContours(third_variant, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # third_variant_cont = plate_lower.copy()
    # # cv2.drawContours(third_variant_cont, cont, -1, (0,255,0), 3)
    # # cv2.imshow("third_variant", third_variant_cont)
    # # cv2.waitKey(0)

    # step_blue = int(900 / 12)
    # step_char7 = int((900 - step_blue) / 7)
    # step_char8 = int((900 - step_blue) / 8)

    # charracters_prob_straight_7 = []
    # charracters_prob_straight_8 = []
    # charracters_prob_upper_7 = []
    # charracters_prob_upper_8 = []
    # charracters_prob_lower_7 = []
    # charracters_prob_lower_8 = []
    # curr_x = step_blue
    # for i in range(7):
    #     charracters_prob_straight_7.append([])
    #     charracters_prob_upper_7.append([])
    #     charracters_prob_lower_7.append([])
    #     img1 = plate_straight[0:600, curr_x:curr_x + step_char7]
    #     img2 = plate_upper[0:600, curr_x:curr_x + step_char7]
    #     img3 = plate_lower[0:600, curr_x:curr_x + step_char7]
    #     img1 = cv2.resize(img1, (225,150))
    #     img2 = cv2.resize(img2, (225,150))
    #     img3 = cv2.resize(img3, (225,150))
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #     img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    #     img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img1 = cv2.erode(img1, kernel, iterations=2)
    #     img2 = cv2.erode(img2, kernel, iterations=2)
    #     img3 = cv2.erode(img3, kernel, iterations=2)
    #     img1 = cv2.dilate(img1, kernel, iterations=2)
    #     img2 = cv2.dilate(img2, kernel, iterations=2)
    #     img3 = cv2.dilate(img3, kernel, iterations=2)
    #     cv2.imshow("first_variant", img1)
    #     cv2.imshow('second_variant', img2)
    #     cv2.imshow('third_variant', img3)
    #     cv2.waitKey(0)
    #     curr_x = curr_x + step_char7         
    #     for char in character_dict:
    #         charracters_prob_straight_7[i].append(cv2.matchTemplate(character_dict[char], img1, cv2.TM_CCOEFF_NORMED))
    #         charracters_prob_upper_7[i].append(cv2.matchTemplate(character_dict[char], img2, cv2.TM_CCOEFF_NORMED))
    #         charracters_prob_lower_7[i].append(cv2.matchTemplate(character_dict[char], img3, cv2.TM_CCOEFF_NORMED))

    # curr_x = step_blue
    # for i in range(8):
    #     charracters_prob_straight_8.append([])
    #     charracters_prob_upper_8.append([])
    #     charracters_prob_lower_8.append([])
    #     img1 = plate_straight[0:600, curr_x:curr_x + step_char8]
    #     img2 = plate_upper[0:600, curr_x:curr_x + step_char8]
    #     img3 = plate_lower[0:600, curr_x:curr_x + step_char8]
    #     img1 = cv2.resize(img1, (225,150))
    #     img2 = cv2.resize(img2, (225,150))
    #     img3 = cv2.resize(img3, (225,150))
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #     img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    #     img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    #     img1 = cv2.erode(img1, kernel, iterations=2)
    #     img2 = cv2.erode(img2, kernel, iterations=2)
    #     img3 = cv2.erode(img3, kernel, iterations=2)
    #     img1 = cv2.dilate(img1, kernel, iterations=2)
    #     img2 = cv2.dilate(img2, kernel, iterations=2)
    #     img3 = cv2.dilate(img3, kernel, iterations=2)
    #     cv2.imshow("first_variant", img1)
    #     cv2.imshow('second_variant', img2)
    #     cv2.imshow('third_variant', img3)
    #     cv2.waitKey(0)
    #     curr_x = curr_x + step_char8        
    #     for char in character_dict:
    #         charracters_prob_straight_8[i].append(cv2.matchTemplate(character_dict[char], img1, cv2.TM_CCOEFF_NORMED))
    #         charracters_prob_upper_8[i].append(cv2.matchTemplate(character_dict[char], img2, cv2.TM_CCOEFF_NORMED))
    #         charracters_prob_lower_8[i].append(cv2.matchTemplate(character_dict[char], img3, cv2.TM_CCOEFF_NORMED))


    # chars = list(character_dict.keys())
    # prob7 = []
    # prob7_chars = []    
    # max_prob = 0
    # max_prob_index = 0
    # for i in range(len(charracters_prob_straight_7)):
    #     max_prob = 0
    #     max_prob_index = 0
    #     for j in range(len(charracters_prob_straight_7[i])):
    #         if charracters_prob_straight_7[i][j] > max_prob:
    #             max_prob = charracters_prob_straight_7[i][j]
    #             max_prob_index = j
    #     for j in range(len(charracters_prob_upper_7[i])):
    #         if charracters_prob_upper_7[i][j] > max_prob:
    #             max_prob = charracters_prob_upper_7[i][j]
    #             max_prob_index = j
    #     for j in range(len(charracters_prob_lower_7[i])):
    #         if charracters_prob_lower_7[i][j] > max_prob:
    #             max_prob = charracters_prob_lower_7[i][j]
    #             max_prob_index = j
    #     prob7.append(max_prob)
    #     prob7_chars.append(chars[max_prob_index])
    # print(prob7_chars)

    # prob8 = []
    # prob8_chars = []
    # max_prob = 0
    # max_prob_index = 0
    # for i in range(len(charracters_prob_straight_8)):
    #     max_prob = 0
    #     max_prob_index = 0
    #     for j in range(len(charracters_prob_straight_8[i])):
    #         if charracters_prob_straight_8[i][j] > max_prob:
    #             max_prob = charracters_prob_straight_8[i][j]
    #             max_prob_index = j
    #             max_prob_set = 0
    #     for j in range(len(charracters_prob_upper_8[i])):
    #         if charracters_prob_upper_8[i][j] > max_prob:
    #             max_prob = charracters_prob_upper_8[i][j]
    #             max_prob_index = j
    #             max_prob_set = 1
    #     for j in range(len(charracters_prob_lower_8[i])):
    #         if charracters_prob_lower_8[i][j] > max_prob:
    #             max_prob = charracters_prob_lower_8[i][j]
    #             max_prob_index = j
    #             max_prob_set = 2
    #     prob8.append(max_prob)
    #     prob8_chars.append(chars[max_prob_index])
    # print(prob8_chars)
    # cv2.imshow("og", img_og)
    # cv2.waitKey(0)

    return 'PO12345'