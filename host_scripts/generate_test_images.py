import os
import glob
from PIL import Image

# Configuration
SOURCE_DIR = r"c:\picoApps\MobileNetTest\test_images"
OUTPUT_FILE = r"c:\picoApps\mobilenet-gtsrb-swifi\firmware\test_images.h"
IMAGES_PER_CLASS = 2 # Change this to 1 or 2 or however many you want
NUM_CLASSES = 43

images_data = []
labels = []
filenames = []

print(f"Scanning {SOURCE_DIR} for images...")

for class_id in range(NUM_CLASSES):
    class_dir = os.path.join(SOURCE_DIR, str(class_id))
    if not os.path.exists(class_dir):
        print(f"Warning: Directory {class_dir} not found.")
        continue
    
    # Get all ppm files in the folder
    ppm_files = sorted(glob.glob(os.path.join(class_dir, "*.ppm")))
    selected_files = ppm_files[:IMAGES_PER_CLASS]
    
    for ppm_file in selected_files:
        try:
            with Image.open(ppm_file) as img:
                # Resize to 32x32 if necessary
                if img.size != (32, 32):
                    img = img.resize((32, 32))
                
                # Ensure it is RGB
                img = img.convert("RGB")
                
                # Get byte data
                pixel_data = list(img.getdata())
                
                # Flatten
                flat_data = []
                for r, g, b in pixel_data:
                    flat_data.extend([r, g, b])
                
                images_data.append(flat_data)
                labels.append(class_id)
                filenames.append(f"{class_id}_{os.path.basename(ppm_file)}") # prepend class_id to filename to prevent duplicates
        except Exception as e:
            print(f"Error processing {ppm_file}: {e}")

num_images = len(images_data)
print(f"Processed {num_images} images. ({IMAGES_PER_CLASS} per class)")

# Generate C Header file
with open(OUTPUT_FILE, "w") as f:
    f.write(f"// Auto-generated test images for GTSRB batch testing\n")
    f.write(f"// Contains {num_images} test images (32x32 RGB) - {IMAGES_PER_CLASS} per class for an unbiased dataset\n\n")
    f.write("#ifndef TEST_IMAGES_H_\n")
    f.write("#define TEST_IMAGES_H_\n\n")
    
    f.write(f"#define NUM_TEST_IMAGES {num_images}\n")
    f.write(f"#define IMAGE_SIZE 3072\n")
    f.write(f"#define STOP_SIGN_CLASS 14\n\n")
    
    # Write filenames
    f.write("const char* image_filenames[] = {\n")
    for i, fname in enumerate(filenames):
        f.write(f'    "{fname}"')
        if i < num_images - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("};\n\n")
    
    # Write labels
    f.write("const int ground_truth_labels[] = {\n    ")
    for i, label in enumerate(labels):
        f.write(f"{label}")
        if i < num_images - 1:
            f.write(", ")
            if (i + 1) % 20 == 0:
                f.write("\n    ")
        else:
            f.write("\n")
    f.write("};\n\n")
    
    # Write image data
    f.write("const unsigned char test_images[][IMAGE_SIZE] = {\n")
    for i, data in enumerate(images_data):
        f.write("    {\n        ")
        for j, val in enumerate(data):
            f.write(f"0x{val:02x}")
            if j < len(data) - 1:
                f.write(", ")
                if (j + 1) % 16 == 0:
                    f.write("\n        ")
        f.write("\n    }")
        if i < num_images - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("};\n\n")
    
    f.write("#endif  // TEST_IMAGES_H_\n")

print(f"Successfully generated {OUTPUT_FILE}!")
