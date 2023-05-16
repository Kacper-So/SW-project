import numpy as np
import cv2
import imutils
import math

def perform_processing(image: np.ndarray, character_dict: dict) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    #processing and contours finding
    img = cv2.resize(image, (800, 600))
    cv2.imshow('photo', img)
    img_og = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            polygon_approx = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
            if len(polygon_approx) == 4:
                ratio = 0
                left_x = min([polygon_approx[0][0][0], polygon_approx[1][0][0], polygon_approx[2][0][0], polygon_approx[3][0][0]])
                right_x = max([polygon_approx[0][0][0], polygon_approx[1][0][0], polygon_approx[2][0][0], polygon_approx[3][0][0]])
                mid_x_th = left_x + (right_x - left_x) / 2
                upper_y = min([polygon_approx[0][0][1], polygon_approx[1][0][1], polygon_approx[2][0][1], polygon_approx[3][0][1]])
                lower_y = max([polygon_approx[0][0][1], polygon_approx[1][0][1], polygon_approx[2][0][1], polygon_approx[3][0][1]])
                mid_y_th = upper_y + (lower_y - upper_y) / 2
                upper_left = [1, 1]
                upper_right = [1, 1]
                lower_left = [1, 1]
                lower_right = [1, 1]
                for point in polygon_approx:
                    if point[0][0] < mid_x_th and point[0][1] < mid_y_th:
                        upper_left = point[0]
                    if point[0][0] > mid_x_th and point[0][1] < mid_y_th:
                        upper_right = point[0]
                    if point[0][0] < mid_x_th and point[0][1] > mid_y_th:
                        lower_left = point[0]
                    if point[0][0] > mid_x_th and point[0][1] > mid_y_th:
                        lower_right = point[0]
                if (pow(upper_right[0], 2) - pow(upper_left[0], 2)) + (pow(upper_right[1], 2) - pow(upper_left[1], 2)) <= 0:
                    plate_lenght = 1
                else:
                    plate_lenght = math.sqrt((pow(upper_right[0], 2) - pow(upper_left[0], 2)) + (pow(upper_right[1], 2) - pow(upper_left[1], 2)))
                if (pow(lower_left[0], 2) - pow(upper_left[0], 2)) + (pow(lower_left[1], 2) - pow(upper_left[1], 2)) <= 0:
                    plate_width = 1
                else:
                    plate_width = math.sqrt((pow(lower_left[0], 2) - pow(upper_left[0], 2)) + (pow(lower_left[1], 2) - pow(upper_left[1], 2)))
                ratio = plate_lenght/ plate_width
                if ratio > 2 and ratio < 5:
                    src = np.float32([[upper_left[0] - 6, upper_left[1] - 6], [upper_right[0] + 6, upper_right[1] - 6], [lower_left[0] - 6, lower_left[1] + 6], [lower_right[0] + 6, lower_right[1] + 6]])
                    dst = np.float32([[0, 0], [1040, 0], [0, 224], [1040, 224]])
                    M = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(img_og, M, (1040, 224))
                    warped_og = warped.copy()
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    warped = cv2.GaussianBlur(warped, (3, 3), 0)
                    ret, warped = cv2.threshold(warped, 90, 255, cv2.THRESH_BINARY)
                    warped = cv2.Canny(warped, 100, 200)
                    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=2)
                    contours, hierarchy = cv2.findContours(warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(warped_og, contours, -1, (0, 255, 0), 1)
                    cv2.imshow('xd', warped_og)
                    cv2.waitKey(0)

                    characters = []
                    characters_x = []
                    characters_w = []
                    keys = list(character_dict.keys())

                    for contour in contours:
                        characters_prob = []
                        if cv2.contourArea(contour) > 3000:
                            x,y,w,h = cv2.boundingRect(contour)
                            if h / w > 1.2 and h / w < 5 and h > 0.5 * 224:
                                char_img = warped_og[y:y+h, x:x+w]
                                char_img = cv2.resize(char_img, (136,225))
                                char_img = cv2.GaussianBlur(char_img, (3, 3), 0)
                                img_og2 = char_img.copy()
                                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                                ret, char_img = cv2.threshold(char_img, 95, 255, cv2.THRESH_BINARY)
                                char_img = cv2.morphologyEx(char_img, cv2.MORPH_OPEN, kernel, iterations=2)
                                # cv2.imshow('xd2',char_img)
                                # cv2.waitKey(0)
                                for char in character_dict:
                                    characters_prob.append(cv2.matchTemplate(character_dict[char], char_img, cv2.TM_CCOEFF_NORMED))
                                max_prob = 0
                                iter = 0
                                iter_max_prob = 0
                                for prob in characters_prob:
                                    if prob > max_prob:
                                        max_prob = prob
                                        iter_max_prob = iter
                                    iter = iter + 1
                                characters.append(keys[iter_max_prob])
                                characters_x.append(x)
                                characters_w.append(w)

                    # print(characters)
                    # print(characters_x)
                    # print(characters_w)
                    combined = list(zip(characters_x, characters_w, characters))
                    sorted_combined = sorted(combined, key=lambda x: x[0])
                    characters_x = [x[0] for x in sorted_combined]
                    characters_w = [x[1] for x in sorted_combined]
                    characters = [x[2] for x in sorted_combined]
                    print(characters)
                    # print(characters_x)
                    # print(characters_w)
                    prev_x = 0
                    prev_w = 0
                    to_del = []
                    for i in range(len(characters)):
                        if prev_x + prev_w > characters_x[i]:
                            to_del.append(i)
                        else:
                            prev_x = characters_x[i]
                            prev_w = characters_w[i]

                    for i in to_del:
                        characters[i] = 'to_del'

                    while 'to_del' in characters:
                        characters.remove('to_del')

                    results.append(characters)
    
    max_len = 0
    index = 0
    for i in range(len(results)):
        if len(results[i]) > max_len:
            max_len = len(results[i])
            index = i

    if max_len == 0:
        result = 'PO4356W'
    else:
        result = ''.join(results[index])

    print(result)
    cv2.waitKey(0)
    
    return result