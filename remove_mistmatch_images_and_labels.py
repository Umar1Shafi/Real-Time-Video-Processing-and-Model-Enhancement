import os

# Define the paths to the image and label directories
image_dir = 'data/images/train'
label_dir = 'data/labels/train'

# List all files in the image and label directories
image_files = set(os.listdir(image_dir))
label_files = set(os.listdir(label_dir))

# Remove file extensions to compare base names
image_basenames = {os.path.splitext(f)[0] for f in image_files}
label_basenames = {os.path.splitext(f)[0] for f in label_files}

# Find unmatched images and labels
unmatched_images = image_files - {f + os.path.splitext(list(image_files)[0])[1] for f in label_basenames}
unmatched_labels = label_files - {f + os.path.splitext(list(label_files)[0])[1] for f in image_basenames}

# Delete unmatched images
for image in unmatched_images:
    os.remove(os.path.join(image_dir, image))
    print(f"Deleted image: {image}")

# Delete unmatched labels
for label in unmatched_labels:
    os.remove(os.path.join(label_dir, label))
    print(f"Deleted label: {label}")

print("Synchronization complete.")