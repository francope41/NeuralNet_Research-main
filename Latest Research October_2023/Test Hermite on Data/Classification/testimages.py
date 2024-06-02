import os

# Starting directory
root_dir = 'valid'


# List to store paths of non-JPG images
non_jpg_images = []

# Traverse through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Get file extension
        _, ext = os.path.splitext(filename)
        # Check if the file extension is not '.jpg'
        if ext.lower() not in ['.jpg', '.jpeg']:  # including '.jpeg' just in case
            non_jpg_path = os.path.join(dirpath, filename)
            non_jpg_images.append(non_jpg_path)
            print(f'Non-JPG Image Found: {non_jpg_path}')

# If you want to save non_jpg_images to a file
with open('non_jpg_images.txt', 'w') as f:
    for path in non_jpg_images:
        f.write(f'{path}\n')
