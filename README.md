# MobileNetV1 GTSRB Fault Injection Experiment

This repository contains the code and scripts needed to replicate the Software Implemented Fault Injection (SWIFI) experiment on a MobileNetV1 model classifying the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The experiment runs on a Raspberry Pi Pico (RP2040/RP2350) using TensorFlow Lite for Microcontrollers (TFLM).

## Experiment Overview

The goal of this experiment is to evaluate the robustness of a quantized MobileNetV1 model against bit-flip errors in its weights. This simulates hardware faults (e.g., radiation-induced single event upsets or voltage drops). 

A host PC orchestrates the campaign by sending serial commands to the Raspberry Pi Pico. The Pico receives the specific tensor, byte offset, and bit to flip, injects the fault directly into its SRAM (where the model is loaded), evaluates a batch of embedded test images, reports the accuracy, and restores the original weight.

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
These files are compiled using the Pico SDK and TFLite Micro framework and flashed to the microcontroller.
* **`main_batch.cpp`**: The primary executable needed. It initializes the model in RAM, listens over USB Serial for `INJECT:<tensor>,<byte>,<bit>` commands, modifies the tensor in RAM, runs inference on all test images, reports the overall accuracy (and specific accuracy for Class 14: Stop Signs), and restores the memory.
* **`main_golden.cpp`**: A clean baseline implementation. It does not contain fault injection logic. It is useful for establishing the base accuracy or using hardware debuggers (like GDB/OpenOCD) to manually halt and inspect state.
* **`model_data.h`**: The actual `.tflite` model converted into a static C-array. It is copied into SRAM at runtime so it can be modified.
* **`test_images.h`**: A subset of the GTSRB dataset converted into flat C-arrays, used for rapid on-device validation without needing an SD card.
* **`model_settings.*`**: Definitions for image input dimensions (32x32 RGB) and category labels (43 traffic sign classes).

### 2. Python Host Scripts
These scripts run on your PC. They require `pyserial`, `numpy`, `tflite`, and `flatbuffers`.
* **`fault_injector.py`**: The "brain" of the fault calculation. It loads the original `.tflite` model, parses the flatbuffer to map tensor indices to exact file/byte offsets, and handles data type conversions (int8, float32, etc.) to figure out exactly which byte and bit need flipping to simulate an error in a specific weight.
* **`run_campaign.py`**: Connects to a single Pico via COM port. It repeatedly queries `fault_injector.py` for random fault parameters, sends the fault to the Pico via Serial, waits for the accuracy result, and logs it to a CSV.
* **`run_campaign_parallel.py` & `run_tensor_sweep.py`**: Multi-threaded versions of the campaign script. They detect multiple Picos plugged into the host and distribute the fault queue across them. `run_tensor_sweep.py` explicitly cycles through every weight tensor in the model to find the most vulnerable layers.
* **`analyze_sweep.py`**: A simple data analysis script that reads the generated CSV results and calculates the percentage of faults that caused critical accuracy degradation.

## Replication Guide

1. **Setup Pico SDK & TFLM**: Ensure you have the Raspberry Pi Pico SDK installed and configured, alongside the `pico-tflmicro` library.
2. **Compile Firmware**: Use CMake to compile `main_batch.cpp`.
   ```bash
   mkdir build && cd build
   cmake ..
   make mobilenet_gtsrb
   ```
3. **Flash the Pico**: Drag and drop the resulting `.uf2` file onto your Raspberry Pi Pico.
4. **Run a Campaign**: Connect the Pico via USB. Install Python dependencies (`pip install pyserial numpy tflite flatbuffers`). Run the sweep:
   ```bash
   python host_scripts/run_tensor_sweep.py
   ```
   *The script will automatically detect the Pico COM port, generate faults, and log results to `campaign_results/`.*
