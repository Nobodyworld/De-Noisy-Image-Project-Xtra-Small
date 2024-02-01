import os
from PIL import Image

input_dir = ".\test_these"
output_dir = ".\test_these\output"
img_height, img_width = 384, 256

def process_image(input_file_path, output_file_path, target_width, target_height):
    # Open the image
    img = Image.open(input_file_path)
    
    # Check if the image is in landscape mode and rotate to portrait if necessary
    if img.width > img.height:
        img = img.transpose(method=Image.Transpose.ROTATE_90)
    
    # Resize the image while maintaining aspect ratio
    img.thumbnail((target_width, target_height))
    
    # Save the processed image
    img.save(output_file_path)

def process_all_images(input_dir, output_dir, target_width, target_height):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        
        # Check if the file is an image
        if os.path.isfile(input_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_file_path = os.path.join(output_dir, file_name)
            process_image(input_file_path, output_file_path, target_width, target_height)

# Process all images in the specified directory
process_all_images(input_dir, output_dir, img_width, img_height)
