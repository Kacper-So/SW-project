import numpy as np
import cv2

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    #Stworzenie słonika z możliwymi znakami oraz ich obrazami
    characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z', 'V', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    img = cv2.imread('font.png')

    height, width, channels = img.shape
    char_spaceing = 25
    part_height = height // 5
    part_width = (width - (7 * char_spaceing)) // 8
    char_img_parts = []
    for i in range(5):
        incremental_char_spaceing = 0
        for j in range(8):
            part = img[i * part_height:(i + 1) * part_height, incremental_char_spaceing + (j * part_width):incremental_char_spaceing + ((j + 1) * part_width)]
            char_img_parts.append(part)
            incremental_char_spaceing = incremental_char_spaceing + char_spaceing

    characters_dict = {}
    for i in range(len(characters)):
        characters_dict[characters[i]] = char_img_parts[i]

    return 'PO12345'
