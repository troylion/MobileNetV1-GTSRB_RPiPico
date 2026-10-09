import csv
import json
import os
import random
import re
import subprocess
import time
import numpy as np

import serial
from serial.tools import list_ports

# Import your injector!
import fault_injector 

# --- CONFIGURATION ---
NUM_FAULTS = 1000
TENSOR_TO_ATTACK = 5
TOTAL_WEIGHTS_IN_TENSOR = 65535
BAUD_RATE = 115200
SERIAL_TIMEOUT_SEC = 120
SERIAL_OPEN_RETRY_SEC = int(os.environ.get("SERIAL_OPEN_RETRY_SEC", "25"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICO_TFLMICRO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BUILD_VARIANT = os.environ.get("PICO_BUILD_VARIANT", "pico1").lower()
BUILD_DIR_NAME = "build_pico2" if BUILD_VARIANT in ["pico2", "rp2350"] else "build"
BOARD_NAME = "pico2 (RP2350)" if BUILD_DIR_NAME == "build_pico2" else "pico (RP2040)"
BUILD_DIR = os.path.join(PICO_TFLMICRO_ROOT, BUILD_DIR_NAME)
UF2_PATH = os.path.join(BUILD_DIR, "examples", "mobilenet_gtsrb", "mobilenet_gtsrb.uf2")

# Tool paths (Pico SDK tools are not on PATH by default)
USERHOME = os.path.expanduser("~")
NINJA_PATH = os.path.join(USERHOME, ".pico-sdk", "ninja", "v1.12.1", "ninja.exe")
PICOTOOL_PATH = os.path.join(USERHOME, ".pico-sdk", "picotool", "2.2.0-a4", "picotool", "picotool.exe")

CAMPAIGN_ID = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "campaign_results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "campaign_checkpoint.json")

RESUME_CAMPAIGN = os.environ.get("RESUME_CAMPAIGN", "1") == "1"
UNIQUE_RESULTS_PER_RUN = os.environ.get("UNIQUE_RESULTS_PER_RUN", "1") == "1"
EXPERIMENT_MODE = int(os.environ.get("EXPERIMENT_MODE", "4"))
SKIP_FLASH = os.environ.get("SKIP_FLASH", "1") == "1"  # Skip compile+flash if firmware already on Pico

# Optional manual override (example: set SERIAL_PORT=COM7)
SERIAL_PORT_OVERRIDE = os.environ.get("SERIAL_PORT", "COM10")


def find_serial_port():
    if SERIAL_PORT_OVERRIDE:
        return SERIAL_PORT_OVERRIDE

    candidates = []
    for port in list_ports.comports():
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if any(token in desc for token in ["pico", "usb serial", "cdc", "serial"]) or "vid:pid" in hwid:
            candidates.append(port.device)

    if candidates:
        return candidates[0]

    ports = list(list_ports.comports())
    if ports:
        return ports[0].device

    return None


def run_cmd(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0


def open_serial_with_retry(preferred_port):
    deadline = time.time() + SERIAL_OPEN_RETRY_SEC
    last_error = None

    while time.time() < deadline:
        candidates = []
        if preferred_port:
            candidates.append(preferred_port)

        auto_port = find_serial_port()
        if auto_port and auto_port not in candidates:
            candidates.append(auto_port)

        for port in candidates:
            try:
                ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT_SEC)
                return ser, port
            except serial.SerialException as exc:
                last_error = exc

        time.sleep(1.0)

    if last_error:
        raise last_error
    raise RuntimeError("No serial port candidates available")


def load_checkpoint():
    if not RESUME_CAMPAIGN:
        return None
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(campaign_id, next_iteration):
    data = {
        "campaign_id": campaign_id,
        "next_iteration": next_iteration,
        "num_faults": NUM_FAULTS,
        "tensor": TENSOR_TO_ATTACK,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_summary_header(path):
    expected = ['Campaign_ID', 'Iteration', 'Tensor', 'Weight_Idx', 'Bit_Flipped', 'Before_Value', 'After_Value', 'Accuracy', 'Stop14_Accuracy']
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    with open(path, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    if not rows:
        return False

    header = rows[0]
    if header == expected:
        return True

    if 'Stop14_Accuracy' in header and 'Before_Value' in header and 'After_Value' in header:
        return True

    rows[0] = expected
    for idx in range(1, len(rows)):
        while len(rows[idx]) < len(expected):
            rows[idx].append('')

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return True


def get_result_paths(campaign_id):
    if UNIQUE_RESULTS_PER_RUN:
        summary = os.path.join(RESULTS_DIR, f"summary_results_{campaign_id}.csv")
        per_image = os.path.join(RESULTS_DIR, f"per_image_results_{campaign_id}.csv")
    else:
        summary = os.path.join(RESULTS_DIR, "summary_results.csv")
        per_image = os.path.join(RESULTS_DIR, "per_image_results.csv")
    return summary, per_image


def flash_pico(uf2_path):
    is_windows = os.name == "nt"
    is_rp2040 = BUILD_DIR_NAME == "build"

    if is_windows and is_rp2040:
        print("[INFO] RP2040 on Windows: using reboot-to-BOOTSEL then normal load")
        if not run_cmd([PICOTOOL_PATH, "reboot", "-f", "-u"]):
            return False
        time.sleep(1.5)
        return run_cmd([PICOTOOL_PATH, "load", uf2_path, "-x"])

    if run_cmd([PICOTOOL_PATH, "load", uf2_path, "-fx"]):
        return True

    print("[WARN] Forced load failed, retrying with normal load...")
    return run_cmd([PICOTOOL_PATH, "load", uf2_path, "-x"])


def query_tensor_size(ser, tensor_idx):
    """Send QUERY command to Pico and return the tensor size in bytes.
    
    Returns the size as an integer, or None if the query failed.
    """
    # Wait for READY_FOR_CMD first
    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        try:
            line = line_bytes.decode('utf-8', errors='replace').strip()
        except UnicodeDecodeError:
            continue
        if line:
            print("Pico:", line)
        if "READY_FOR_CMD" in line:
            break

    cmd_str = f"QUERY:{tensor_idx}\n"
    print(f"[INFO] Querying tensor {tensor_idx} size...")
    ser.write(cmd_str.encode('utf-8'))

    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        try:
            line = line_bytes.decode('utf-8', errors='replace').strip()
        except UnicodeDecodeError:
            continue
        if line:
            print("Pico:", line)
        if line.startswith("SIZE:"):
            # SIZE:<tensor_idx>,<bytes>
            parts = line.split(":")[1].split(",")
            if len(parts) == 2:
                return int(parts[1])
        if line.startswith("ERROR:"):
            print(f"[WARN] Tensor query failed: {line}")
            return None


def set_pico_mode(ser, mode):
    """Send MODE:VERBOSE or MODE:QUIET command to Pico.
    
    Waits for READY_FOR_CMD first, then sends the mode command.
    """
    # Wait for READY_FOR_CMD
    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        try:
            line = line_bytes.decode('utf-8', errors='replace').strip()
        except UnicodeDecodeError:
            continue
        if line:
            print("Pico:", line)
        if "READY_FOR_CMD" in line:
            break

    cmd_str = f"MODE:{mode}\n"
    print(f"[INFO] Setting Pico to {mode} mode...")
    ser.write(cmd_str.encode('utf-8'))

    # Read acknowledgement
    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        try:
            line = line_bytes.decode('utf-8', errors='replace').strip()
        except UnicodeDecodeError:
            continue
        if line:
            print("Pico:", line)
        if f"MODE:{mode}" in line:
            print(f"[INFO] Pico confirmed {mode} mode.")
            return True
        if "ERROR:" in line:
            print(f"[WARN] Mode switch failed: {line}")
            return False


def compute_before_after_values(raw_bytes, byte_offset, bit_in_byte, dtype):
    """Compute before/after values for the faulted weight, handling multi-byte dtypes.
    
    For int8 (itemsize=1), reads 1 byte.
    For float32 (itemsize=4), reads 4 bytes and interprets as float.
    """
    itemsize = np.dtype(dtype).itemsize
    
    # Find the start of the weight that contains byte_offset
    weight_start = (byte_offset // itemsize) * itemsize
    
    # Read the full weight value (may be 1, 2, or 4 bytes)
    orig_weight_bytes = bytearray(raw_bytes[weight_start : weight_start + itemsize])
    before_value = np.frombuffer(bytes(orig_weight_bytes), dtype=dtype)[0]
    
    # Apply the bit flip to compute the "after" value
    faulted_weight_bytes = bytearray(orig_weight_bytes)
    byte_within_weight = byte_offset - weight_start
    faulted_weight_bytes[byte_within_weight] ^= (1 << bit_in_byte)
    after_value = np.frombuffer(bytes(faulted_weight_bytes), dtype=dtype)[0]
    
    return before_value, after_value


print("\n========== CAMPAIGN CONFIG ==========")
print(f"Board target      : {BOARD_NAME}")
print(f"Build folder      : {BUILD_DIR}")
print(f"UF2 path          : {UF2_PATH}")
print(f"Serial override   : {SERIAL_PORT_OVERRIDE if SERIAL_PORT_OVERRIDE else 'AUTO-DETECT'}")
checkpoint = load_checkpoint()
if checkpoint and checkpoint.get("num_faults") == NUM_FAULTS and checkpoint.get("tensor") == TENSOR_TO_ATTACK:
    CAMPAIGN_ID = checkpoint.get("campaign_id", CAMPAIGN_ID)
    START_ITERATION = int(checkpoint.get("next_iteration", 0))
else:
    START_ITERATION = 0

# --- MANUAL OVERRIDE (uncomment to force a specific campaign) ---
# CAMPAIGN_ID = "20260426_194556"
# START_ITERATION = 0 
# -----------------------

print(f"Num faults        : {NUM_FAULTS}")
print(f"Tensor to attack  : {TENSOR_TO_ATTACK}")
print(f"Results folder    : {RESULTS_DIR}")
print(f"Campaign ID       : {CAMPAIGN_ID}")
print(f"Resume enabled    : {RESUME_CAMPAIGN}")
print(f"Unique files/run  : {UNIQUE_RESULTS_PER_RUN}")
print(f"Experiment Mode   : {EXPERIMENT_MODE}")
print(f"Skip flash        : {SKIP_FLASH}")
print(f"Start iteration   : {START_ITERATION + 1}")
print("=====================================\n")

os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_CSV_PATH, PER_IMAGE_CSV_PATH = get_result_paths(CAMPAIGN_ID)

per_image_pattern = re.compile(
    r"^\[\s*(\d+)\s*/\s*(\d+)\]\s+(\S+)\s+GT:\s*(-?\d+)\s+Pred:\s*(-?\d+)\s+Conf:\s*(-?\d+)\s+\(\s*(\d+)%\)\s+(OK|WRONG)\s+\((\d+)\s+ms\)$"
)
stop14_pattern = re.compile(r"^Class\s+14:\s+\d+\s*/\s*\d+\s+\(([^)]+)\)")
stop14_new_pattern = re.compile(r"^STOP14:\s+\d+/\d+\s+\(([^)]+)\)")

summary_exists = os.path.exists(SUMMARY_CSV_PATH) and os.path.getsize(SUMMARY_CSV_PATH) > 0
per_image_exists = os.path.exists(PER_IMAGE_CSV_PATH) and os.path.getsize(PER_IMAGE_CSV_PATH) > 0

if summary_exists:
    ensure_summary_header(SUMMARY_CSV_PATH)

with open(SUMMARY_CSV_PATH, mode='a', newline='') as summary_file, open(PER_IMAGE_CSV_PATH, mode='a', newline='') as per_image_file:
    summary_writer = csv.writer(summary_file)
    per_image_writer = csv.writer(per_image_file)
    next_iteration = START_ITERATION

    if not summary_exists:
        summary_writer.writerow(['Campaign_ID', 'Iteration', 'Tensor', 'Weight_Idx', 'Bit_Flipped', 'Before_Value', 'After_Value', 'Accuracy', 'Stop14_Accuracy'])
        summary_file.flush()

    if not per_image_exists:
        per_image_writer.writerow([
            'Campaign_ID',
            'Fault_Iteration',
            'Image_Index',
            'Image_Total',
            'Filename',
            'GT_Class',
            'Pred_Class',
            'Confidence_Raw',
            'Confidence_Percent',
            'Status',
            'Inference_Time_ms'
        ])
        per_image_file.flush()

    try:
        print("[INFO] Initializing RNG and loading model unconditionally...")
        # Init random state
        if getattr(fault_injector, "RANDOM_SEED", None) is not None and not fault_injector._SEED_INITIALIZED:
            np.random.seed(fault_injector.RANDOM_SEED)
            import random
            random.seed(fault_injector.RANDOM_SEED)
            fault_injector._SEED_INITIALIZED = True
        
        # Load model once to get valid indices and raw_bytes buffer to draw from
        buf, model = fault_injector.load_model(fault_injector.MODEL_PATH)
        arr, raw_bytes = fault_injector.get_tensor_data_as_numpy(model, buf, TENSOR_TO_ATTACK, fault_injector.DTYPE)
        itemsize = np.dtype(fault_injector.DTYPE).itemsize
        total_bits = itemsize * 8
        arr_flat = arr.flatten()
        
        if EXPERIMENT_MODE == 1 or EXPERIMENT_MODE == 2:
            valid_indices = np.where(arr_flat > 0)[0]
        elif EXPERIMENT_MODE == 3:
            valid_indices = np.where(arr_flat < 0)[0]
        else:
            valid_indices = np.arange(len(raw_bytes) // itemsize)

        if START_ITERATION > 0:
            print(f"[INFO] Fast-forwarding RNG state by exactly {START_ITERATION} iterations...")
            for _ in range(START_ITERATION):
                _ = np.random.choice(valid_indices)
                if EXPERIMENT_MODE == 1:
                    _ = np.random.randint(0, total_bits - 1)
                elif EXPERIMENT_MODE == 2 or EXPERIMENT_MODE == 3:
                    pass
                else:
                    _ = np.random.randint(0, total_bits)
            print("[INFO] RNG state restored.")

        # --- 1 & 2: COMPILE AND FLASH ONCE ---
        if SKIP_FLASH:
            print("[INFO] SKIP_FLASH=1: Skipping compile and flash (firmware already on Pico)")
            # Still reboot the Pico to get a clean USB CDC connection
            print("[INFO] Rebooting Pico to reset serial connection...")
            run_cmd([PICOTOOL_PATH, "reboot", "-f"])
            time.sleep(4)
        else:
            print("Compiling .uf2...")
            if not run_cmd([NINJA_PATH, "-C", BUILD_DIR, "examples/mobilenet_gtsrb/all"]):
                print("[ERROR] Build failed. Stopping campaign.")
                raise RuntimeError("Build failed")

            if not os.path.exists(UF2_PATH):
                print(f"[ERROR] UF2 not found at: {UF2_PATH}")
                raise RuntimeError("UF2 file not found")

            print("Flashing Pico...")
            if not flash_pico(UF2_PATH):
                print("[ERROR] Flash failed. Stopping campaign.")
                raise RuntimeError("Flash failed")

            # Give USB CDC a moment to enumerate after flash/reset
            time.sleep(4)

        # --- 3: OPEN SERIAL PORT ONCE ---
        print("Listening to Pico...")
        serial_port = find_serial_port()
        if not serial_port:
            print("[ERROR] No serial port found. Set SERIAL_PORT env var and retry.")
            raise RuntimeError("No serial port")

        print(f"Using serial port: {serial_port}")
        try:
            ser, active_port = open_serial_with_retry(serial_port)
        except Exception as exc:
            print(f"[ERROR] Could not open serial port after retry window: {exc}")
            print("[INFO] Tip: confirm port in Device Manager or set SERIAL_PORT explicitly.")
            raise

        if active_port != serial_port:
            print(f"[INFO] Port switched to {active_port}")

        # --- 3.5: SYNC, QUERY, AND SET QUIET MODE ---
        # Flush any stale data and sync with the Pico
        print("[INFO] Syncing with Pico...")
        ser.reset_input_buffer()
        time.sleep(0.5)

        # Send a junk command to consume the current READY_FOR_CMD
        # and trigger a fresh ERROR + READY_FOR_CMD cycle
        ser.write(b"SYNC\n")
        synced = False
        sync_deadline = time.time() + 15
        while time.time() < sync_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                continue
            if line:
                print(f"  Pico: {line}")
            if "READY_FOR_CMD" in line:
                synced = True
                break

        if not synced:
            print("[ERROR] Could not sync with Pico (no READY_FOR_CMD received)")
            raise RuntimeError("Pico sync failed")

        print("[INFO] Pico synced successfully!")

        # Query tensor size
        print(f"[INFO] Querying tensor {TENSOR_TO_ATTACK} size...")
        ser.write(f"QUERY:{TENSOR_TO_ATTACK}\n".encode('utf-8'))
        tensor_size = None
        query_deadline = time.time() + 10
        while time.time() < query_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                continue
            if line:
                print(f"  Pico: {line}")
            if line.startswith("SIZE:"):
                parts = line.split(":")[1].split(",")
                if len(parts) == 2:
                    tensor_size = int(parts[1])
                break
            if line.startswith("ERROR:"):
                print(f"[WARN] Tensor query failed: {line}")
                break

        if tensor_size is not None:
            print(f"[INFO] Tensor {TENSOR_TO_ATTACK} size confirmed by Pico: {tensor_size:,} bytes")
            if tensor_size != len(raw_bytes):
                print(f"[WARN] Local raw_bytes length ({len(raw_bytes)}) != Pico tensor size ({tensor_size})")
        else:
            print(f"[WARN] Could not query tensor size. Using local estimate: {len(raw_bytes)} bytes")

        # Wait for next READY_FOR_CMD before setting mode
        ready_deadline = time.time() + 10
        while time.time() < ready_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                continue
            if "READY_FOR_CMD" in line:
                break

        # Set QUIET mode for faster serial throughput during campaign
        print("[INFO] Setting QUIET mode...")
        ser.write(b"MODE:QUIET\n")
        mode_deadline = time.time() + 5
        while time.time() < mode_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            try:
                line = line_bytes.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                continue
            if line:
                print(f"  Pico: {line}")
            if "MODE:QUIET" in line:
                print("[INFO] Quiet mode confirmed.")
                break

        # --- 4: MAIN INJECTION LOOP ---
        campaign_start = time.time()
        for i in range(START_ITERATION, NUM_FAULTS):
            fault_start = time.time()
            elapsed_total = fault_start - campaign_start
            if i > START_ITERATION:
                avg_per_fault = elapsed_total / (i - START_ITERATION)
                remaining = avg_per_fault * (NUM_FAULTS - i)
                eta_str = f"  ETA: {remaining/60:.0f}m {remaining%60:.0f}s"
            else:
                eta_str = ""
            print(f"\n====== FAULT {i+1}/{NUM_FAULTS} [{elapsed_total:.0f}s elapsed]{eta_str} ======")

            # Generate Fault Parameters matching fault_injector
            rand_weight = int(np.random.choice(valid_indices))
            
            if EXPERIMENT_MODE == 1:
                actual_bit = int(np.random.randint(0, total_bits - 1))
            elif EXPERIMENT_MODE == 2 or EXPERIMENT_MODE == 3:
                actual_bit = total_bits - 1
            else:
                actual_bit = int(np.random.randint(0, total_bits))

            byte_offset = rand_weight * itemsize + (actual_bit // 8)
            bit_in_byte = actual_bit % 8

            # Calculate before and after values (handles any dtype: int8, float32, etc.)
            before_value, after_value = compute_before_after_values(
                raw_bytes, byte_offset, bit_in_byte, fault_injector.DTYPE
            )
            # Convert to Python native types for CSV serialization
            before_value = float(before_value) if itemsize > 1 else int(before_value)
            after_value = float(after_value) if itemsize > 1 else int(after_value)

            # Wait for READY_FOR_CMD
            print("Waiting for Pico to be ready...")
            while True:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                try:
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    print("Pico:", line)
                if "READY_FOR_CMD" in line:
                    break
                    
            # Send injection command over serial
            cmd_str = f"INJECT:{TENSOR_TO_ATTACK},{byte_offset},{bit_in_byte}\n"
            print(f"  -> INJECT tensor={TENSOR_TO_ATTACK} byte={byte_offset} bit={bit_in_byte} (weight[{rand_weight}] bit {actual_bit})")
            print(f"     Before={before_value} After={after_value}")
            ser.write(cmd_str.encode('utf-8'))

            # Read back results
            final_accuracy = "ERROR"
            stop14_accuracy = "N/A"

            while True:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                try:
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                except UnicodeDecodeError:
                    continue
                    
                if line:
                    print("Pico:", line)

                match = per_image_pattern.match(line)
                if match:
                    per_image_writer.writerow([
                        CAMPAIGN_ID,
                        i + 1,
                        int(match.group(1)),
                        int(match.group(2)),
                        match.group(3),
                        int(match.group(4)),
                        int(match.group(5)),
                        int(match.group(6)),
                        int(match.group(7)),
                        match.group(8),
                        int(match.group(9)),
                    ])
                    per_image_file.flush()
            
                # Catch the accuracy and stop accuracy
                if "Accuracy:" in line:
                    parts = line.split("Accuracy:")
                    if len(parts) > 1:
                        final_accuracy = parts[1].strip()

                stop_match = stop14_pattern.match(line) or stop14_new_pattern.match(line)
                if stop_match:
                    stop14_accuracy = stop_match.group(1)

                # Break the loop when the Pico signals it is finished
                if "TEST_COMPLETE" in line:
                    break

            # 6. RECORD DATA
            summary_writer.writerow([
                CAMPAIGN_ID,
                i + 1,
                TENSOR_TO_ATTACK,
                rand_weight,
                actual_bit,
                before_value,
                after_value,
                final_accuracy,
                stop14_accuracy,
            ])
            summary_file.flush()
            next_iteration = i + 1
            save_checkpoint(CAMPAIGN_ID, next_iteration)
            fault_elapsed = time.time() - fault_start
            print(f"  -> RESULT: Acc={final_accuracy}  Stop14={stop14_accuracy}  ({fault_elapsed:.1f}s)")

    except RuntimeError as re:
        print(f"\n[INFO] Campaign halted: {re}")
    except KeyboardInterrupt:
        print("\n[INFO] Campaign interrupted by user. Resume is available on next run.")
    finally:
        try:
            if 'ser' in locals() and ser.is_open:
                ser.close()
        except Exception:
            pass

    print("\nCAMPAIGN COMPLETE!")
    print(f"Summary CSV   : {SUMMARY_CSV_PATH}")
    print(f"Per-image CSV : {PER_IMAGE_CSV_PATH}")
    if next_iteration >= NUM_FAULTS and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[INFO] All iterations completed. Checkpoint cleared.")
    elif os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Incomplete run. Resume from iteration {next_iteration + 1} next time.")