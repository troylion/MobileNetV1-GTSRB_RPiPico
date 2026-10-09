# MobileNetV1 GTSRB Fault Injection Experiment

This repository contains the code and scripts needed to replicate the Software Implemented Fault Injection (SWIFI) experiment on a MobileNetV1 model classifying the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The experiment runs on a Raspberry Pi Pico (RP2040) using TensorFlow Lite for Microcontrollers (TFLM).

*Note: This experiment currently works on the original Raspberry Pi Pico (RP2040). It has not yet been tested on the Raspberry Pi Pico 2 (RP2350).*

## Experiment Overview

The goal of this experiment is to evaluate the robustness of a quantized MobileNetV1 model against bit-flip errors in its weights. This simulates hardware faults (e.g., radiation-induced single event upsets or voltage drops). 

A host PC orchestrates the campaign by sending serial commands to the Raspberry Pi Pico. The Pico receives the specific tensor, byte offset, and bit to flip, injects the fault directly into its SRAM (where the model is loaded), evaluates a batch of embedded test images, reports the accuracy, and restores the original weight.

## What is SWIFI?

SWIFI stands for **Software-Implemented Fault Injection**. It is a technique used to evaluate how a system (like a neural network) behaves when hardware faults occur, without needing expensive physical fault injection equipment like lasers or radiation beams. Instead of physically causing a bit-flip in the hardware, the software artificially modifies a value in memory (SRAM) to simulate the effect of a hardware fault. This allows for precise, repeatable, and automated testing of a system's vulnerability to errors like Single Event Upsets (SEUs).

## Repository Structure

If you are setting up this repository from scratch, your directory should be organized as follows:

```text
├── CMakeLists.txt                # CMake build configuration for the Pico firmware
├── firmware/
│   ├── main_batch.cpp            # Primary SWIFI firmware (listens for faults & runs batch evaluation)
│   ├── main_golden.cpp           # Baseline firmware (clean inference without faults)
│   ├── model_settings.cpp        # Model configurations and GTSRB class labels
│   ├── model_settings.h          # Header for model dimensions and settings
│   ├── model_data.h              # The compiled TFLite model weights as a C-array
│   └── test_images.h             # Embedded GTSRB test images and ground truth labels
└── host_scripts/
    ├── fault_injector.py         # Core utility to parse the TFLite flatbuffer and compute fault offsets
    ├── run_campaign.py           # Script to run a fault injection campaign on a single Pico
    ├── run_campaign_parallel.py  # Script to run a fault injection campaign using multiple Picos
    ├── run_tensor_sweep.py       # Sweeps faults across ALL weight tensors to map vulnerabilities
    └── analyze_sweep.py          # Parses the resulting CSVs and generates a summary/heat map
```

## File Explanations & Requirements

### 1. C++ Firmware (Raspberry Pi Pico)
*Role: This is the base program running on the microcontroller that tests the embedded images against the neural network model.*

These files are compiled using the Pico SDK and TFLite Micro framework and flashed to the microcontroller.
* **`main_batch.cpp`**: The primary executable needed. It initializes the model in RAM, listens over USB Serial for `INJECT:<tensor>,<byte>,<bit>` commands, modifies the tensor in RAM, runs inference on all test images, reports the overall accuracy (and specific accuracy for Class 14: Stop Signs), and restores the memory.
* **`main_golden.cpp`**: A clean baseline implementation. It does not contain fault injection logic. It is useful for establishing the base accuracy or using hardware debuggers (like GDB/OpenOCD) to manually halt and inspect state.
* **`model_data.h`**: The actual `.tflite` model converted into a static C-array. It is copied into SRAM at runtime so it can be modified.
* **`test_images.h`**: A subset of the GTSRB dataset converted into flat C-arrays, used for rapid on-device validation without needing an SD card.
* **`model_settings.*`**: Definitions for image input dimensions (32x32 RGB) and category labels (43 traffic sign classes).

### 2. Python Host Scripts
*Role: These scripts orchestrate the fault injections. They run on your PC and send the targeted bit-flip commands to the Pico over USB Serial.*

These scripts require `pyserial`, `numpy`, `tflite`, and `flatbuffers`.
* **`fault_injector.py`**: The "brain" of the fault calculation. It loads the original `.tflite` model, parses the flatbuffer to map tensor indices to exact file/byte offsets, and handles data type conversions (int8, float32, etc.) to figure out exactly which byte and bit need flipping to simulate an error in a specific weight.
* **`run_campaign.py`**: Connects to a single Pico via COM port. It repeatedly queries `fault_injector.py` for random fault parameters, sends the fault to the Pico via Serial, waits for the accuracy result, and logs it to a CSV.
* **`run_campaign_parallel.py` & `run_tensor_sweep.py`**: Multi-threaded versions of the campaign script. They detect multiple Picos plugged into the host and distribute the fault queue across them. `run_tensor_sweep.py` explicitly cycles through every weight tensor in the model to find the most vulnerable layers.
* **`analyze_sweep.py`**: A simple data analysis script that reads the generated CSV results and calculates the percentage of faults that caused critical accuracy degradation.

## Modifying the Code for Different Fault Models

If you want to emulate different types of upsets (such as multiple bit upsets, stuck-at faults, or zeroing out entire filters), you will need to adjust the following files:

1. **`host_scripts/fault_injector.py`**: This script generates the fault parameters. It already contains skeleton functions for `fault_zero_filter`, `fault_noise`, and `fault_clamp`. You can modify the random generation logic to compute parameters for your specific fault model.
2. **`host_scripts/run_campaign.py`**: Modify the payload sent via the `INJECT` command if your new fault model requires more parameters (e.g., sending a specific mask or value instead of just a bit position).
3. **`firmware/main_batch.cpp`**: This is where the fault is actually applied in RAM. Locate the section that handles the `INJECT:` command. Currently, it uses an XOR operation to flip a single bit (`fault_buffer[byte_within] ^= (1 << bit_in_byte);`). You would change this logical operation depending on your fault. For example, to simulate a stuck-at-0 fault, you might use a bitwise AND (`fault_buffer[byte_within] &= ~(1 << bit_in_byte);`).

## Swapping the Model or Images (C-Byte Arrays)

Microcontrollers do not have traditional file systems. Instead, the `.tflite` model and the testing images must be converted into C-byte arrays and compiled directly into the binary firmware. 

- **`model_data.h`**: Contains the complete `.tflite` MobileNetV1 model serialized as an `unsigned char` array (e.g., `const unsigned char model_data[] = { 0x1c, 0x00, ... };`). The firmware reads this array to initialize the neural network in SRAM. 
  *(Origin: The included MobileNetV1 model was trained from scratch on the GTSRB dataset using Google Colab. It was then converted into a `.tflite` file, and finally converted into this C-byte array header file using the `xxd` command in Git Bash).*
- **`test_images.h`**: Contains the GTSRB test images and their ground-truth labels. The images are stored as flat 1D arrays of bytes, representing the raw pixel data.

### How to use a different model or dataset:
If you want to run this experiment on a different neural network (like ResNet) or a different dataset (like CIFAR-10), you can simply swap out these header files:
1. Train and quantize your new `.tflite` model.
2. Use a command line tool like `xxd` to convert your `.tflite` file into a C-array (e.g., `xxd -i my_new_model.tflite > model_data.h`). *(Note: If you are on Windows, `xxd` is not a native command in CMD or PowerShell. You will need to run this command in **Git Bash** or WSL).*
3. Convert your new test images into a similar C-array format and replace `test_images.h`.
4. Update `model_settings.h` to reflect the new image dimensions, number of channels, and category count.
5. Recompile your firmware using CMake and flash the new `.uf2` file to the Pico!

## Replication Guide

### 1. Setup Pico SDK & TFLM
Ensure you have the Raspberry Pi Pico SDK installed and configured, alongside the `pico-tflmicro` library.

### 2. Compile Firmware (Creating .uf2 and .elf files)
To build the project and generate the executable files, use CMake:
```bash
# Create a build directory
mkdir build && cd build

# Configure CMake (make sure your PICO_SDK_PATH is set)
cmake ..

# Build the executable
make mobilenet_gtsrb
```
This process will generate several files in the `build/` directory, including:
- **`mobilenet_gtsrb.elf`**: The executable linked file, useful for debugging with GDB or OpenOCD.
- **`mobilenet_gtsrb.uf2`**: The USB Flashing Format file, used to easily program the Pico over USB.

### 3. Flash and Run on the Raspberry Pi Pico
1. While unplugged, hold down the **BOOTSEL** button on your Raspberry Pi Pico.
2. While continuing to hold BOOTSEL, plug the Pico into your computer's USB port.
3. Release the BOOTSEL button. The Pico will mount as a USB Mass Storage Device named `RPI-RP2` (the default for the RP2040).
4. Drag and drop the `mobilenet_gtsrb.uf2` file onto the `RPI-RP2` drive.
5. The Pico will automatically disconnect, reboot, and immediately start running the firmware.

### 4. Run a SWIFI Campaign (Using the Provided Flatbuffer Fault Injection)
*Note: If you are using an alternative fault injection method (such as GDB/debugger-based injection, or hardcoding upsets directly in the C++ firmware), you will not run these Python campaign scripts. You will simply flash your firmware (like `main_golden.cpp`) and interact with it via your debugger or serial monitor.*

If you are using the flatbuffer-based SWIFI method included in this repository, once the Pico is running, connect to it using the host scripts. Install the Python dependencies:
```bash
pip install pyserial numpy tflite flatbuffers
```
Run the automated tensor sweep:
```bash
python host_scripts/run_tensor_sweep.py
```
*The script will automatically detect the Pico COM port, generate faults, and log the accuracy results to the `campaign_results/` directory.*
