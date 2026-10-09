import os

def convert_tflite_to_c_array(tflite_path, output_paths, array_name):
    print(f"Reading {tflite_path}...")
    with open(tflite_path, 'rb') as f:
        data = f.read()

    for path in output_paths:
        print(f"Writing to {path}...")
        with open(path, 'w') as f:
            f.write(f'// Auto-generated TensorFlow Lite model data\n')
            f.write(f'// Source: {os.path.basename(tflite_path)}\n\n')
            f.write(f'#ifndef {array_name.upper()}_H\n')
            f.write(f'#define {array_name.upper()}_H\n\n')
            f.write(f'#include <cstdint>\n\n')
            f.write(f'alignas(16) const unsigned char {array_name}[] = {{\n')
            
            for i, byte in enumerate(data):
                if i % 12 == 0:
                    f.write('  ')
                f.write(f'0x{byte:02x}, ')
                if (i + 1) % 12 == 0:
                    f.write('\n')
                    
            f.write(f'\n}};\n\n')
            f.write(f'const int {array_name}_len = {len(data)};\n\n')
            f.write(f'#endif // {array_name.upper()}_H\n')

if __name__ == "__main__":
    tflite_file = r"c:\picoApps\MobileNetTest\mobilenet_gtsrb_UNBIASED.tflite"
    outputs = [
        r"c:\picoApps\mobilenet-gtsrb-swifi\firmware\model_data_unbiased.h",
        r"c:\picoApps\MobileNetTest\model_data_unbiased.h"
    ]
    
    if os.path.exists(tflite_file):
        convert_tflite_to_c_array(tflite_file, outputs, "model_data_unbiased")
        print("Done!")
    else:
        print(f"Error: Could not find {tflite_file}")
