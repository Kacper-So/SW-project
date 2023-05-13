import argparse
import json
from pathlib import Path
import cv2

from processing.utils import perform_processing


def main():

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



    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image, characters_dict)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
