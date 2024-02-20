import cv2
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse

def flip(images, rotation_degree=180):
    """
    Rotate images 180 degrees.
    """
    rotated_images = []
    for img_path in tqdm(images, desc="Rotating images"):
        img = cv2.imread(img_path)
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        rotated_images.append(rotated_img)
    return rotated_images

def resize(images, width=1280, height=800):
    """
    Resize images to the specified width and height.
    """
    resized_images = []
    for img in tqdm(images, desc="Resizing images"):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_resized = img_pil.resize((width, height), Image.ANTIALIAS)
        img_resized_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        resized_images.append(img_resized_cv)
    return resized_images

def find_non_image_region_circle(example_image_path):
    """
    Find the bounding circle of the non-image region, accounting for top and bottom cropping.
    """
    example_image = cv2.imread(example_image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(example_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(x), int(y), int(radius))

def mask_images(images, circle_center, circle_radius):
    """
    Apply a circular mask with top and bottom cropping to images.
    """
    masked_images = []
    for img in tqdm(images, desc="Applying masks"):
        img_h, img_w = img.shape[:2]
        center = (img_w // 2, img_h // 2)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(mask, center, circle_radius, (255, 255, 255), thickness=-1)
        cropped_height_top = circle_center[1] - circle_radius
        cropped_height_bottom = img_h - (circle_center[1] + circle_radius)
        if cropped_height_top > 0:
            mask[:cropped_height_top, :] = 0
        if cropped_height_bottom > 0:
            mask[-cropped_height_bottom:, :] = 0
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        masked_images.append(masked_img)
    return masked_images

def process_images(input_folder, output_folder, example_image_path, rotation_degree, width, height):
    """
    Process all images in the input folder: rotate, resize, and apply a mask.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
    print(f"Found {len(images_paths)} images to process.")
    
    fliped_images = flip(images_paths, rotation_degree)
    resized_images = resize(fliped_images, width, height)
    x, y, radius = find_non_image_region_circle(example_image_path)
    masked_images = mask_images(resized_images, (x, y), radius)

    for i, img in enumerate(tqdm(masked_images, desc="Saving processed images")):
        output_path = os.path.join(output_folder, os.path.basename(images_paths[i]))
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images: rotate, resize, and mask.")
    parser.add_argument("--input_folder", help="Input folder containing images to process")
    parser.add_argument("--output_folder", help="Output folder for processed images")
    parser.add_argument("--example_image_path", help="Path to the example image for masking reference", default = 'videos/xrego_test1/male_006_a_a.rgba.000001.png' )
    parser.add_argument("--rotation_degree", type=int, default=180, help="Degrees to rotate images (default: 180)")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize images to (default: 1280)")
    parser.add_argument("--height", type=int, default=800, help="Height to resize images to (default: 800)")
    
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder, args.example_image_path,
                   args.rotation_degree, args.width, args.height)
