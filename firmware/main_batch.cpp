/* 
 * Author: Kristian Bailey, North Carolina A&T State University
 * Project: MobileNet GTSRB SWIFI Framework
 * Description: Tests multiple embedded images and calculates accuracy.
 *              Supports host-controlled SWIFI (Software Implemented Fault Injection)
 *              via USB serial commands.
 */
#include "pico/stdlib.h"
#include <cstdio>
#include <cstring>

#include "model_data.h"
#include "model_settings.h"
#include "test_images.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals
namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// RAM buffer for storing a single tensor to be faulted (up to ~66KB (covers
// largest tensor: 65536 bytes))
uint8_t fault_buffer[66000] __attribute__((aligned(16)));
void *original_data_ptr = nullptr;
int faulted_tensor_idx = -1;

// Verbose mode: when false, skips per-image output for faster serial throughput
bool verbose_mode = true;

// Total number of tensors in the model (set after model is loaded)
int total_tensors = 0;

// ─── GDB Fault Slots ───
// Write these via GDB to inject a fault before the next auto-baseline run.
// After the batch completes, gdb_fault_tensor is reset to -1.
//
// GDB usage:
//   (gdb) set gdb_fault_tensor = 5
//   (gdb) set gdb_fault_byte = 1000
//   (gdb) set gdb_fault_bit = 3
//   (gdb) continue
//
volatile int gdb_fault_tensor = -1;  // -1 = no fault (clean run)
volatile int gdb_fault_byte = 0;
volatile int gdb_fault_bit = 0;

} // namespace

struct InferenceResult {
  int predicted_class;
  int8_t confidence; // Raw int8 score (-128 to 127)
};

InferenceResult run_inference(const unsigned char *image_data) {
  // Copy image to input tensor
  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < 3072; i++) {
      input->data.int8[i] = (int8_t)(image_data[i] - 128);
    }
  } else {
    memcpy(input->data.uint8, image_data, 3072);
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    return {-1, 0};
  }

  // Get result - find max score and index
  int8_t *scores = output->data.int8;
  int max_idx = 0;
  int8_t max_score = scores[0];
  for (int i = 1; i < kCategoryCount; i++) {
    if (scores[i] > max_score) {
      max_score = scores[i];
      max_idx = i;
    }
  }
  return {max_idx, max_score};
}

// Run all test images and print results
void run_batch_test() {
  int correct = 0;
  int total = 0;
  uint32_t total_time = 0;

  int class_correct[kCategoryCount] = {0};
  int class_total[kCategoryCount] = {0};

  if (verbose_mode) {
    printf("\n##############################################\n");
    printf("  STARTING BATCH\n");
    printf("##############################################\n");
  }

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    int gt_class = ground_truth_labels[i];

    uint32_t start = time_us_32();
    InferenceResult result = run_inference(test_images[i]);
    uint32_t elapsed = time_us_32() - start;
    total_time += elapsed;

    int pred_class = result.predicted_class;
    int8_t raw_score = result.confidence;
    int confidence_pct = (raw_score + 128) * 100 / 255;

    bool is_correct = (pred_class == gt_class);
    if (is_correct) {
      correct++;
      class_correct[gt_class]++;
    }
    total++;
    class_total[gt_class]++;

    if (verbose_mode) {
      const char *status = is_correct ? "OK" : "WRONG";
      printf("[%3d/%d] %s GT:%2d Pred:%2d Conf:%4d (%3d%%) %s (%lu ms)\n",
             i + 1, NUM_TEST_IMAGES, image_filenames[i], gt_class, pred_class,
             raw_score, confidence_pct, status, elapsed / 1000);
    }
  }

  // Summary (always printed)
  printf("\n");
  printf("==============================================\n");
  printf("  OVERALL RESULTS\n");
  printf("==============================================\n");
  printf("Correct: %d / %d\n", correct, total);
  printf("Accuracy: %.1f%%\n", (float)correct / total * 100.0f);
  printf("Avg inference time: %.1f ms\n",
         (float)total_time / total / 1000.0f);
  printf("Total time: %.1f seconds\n", (float)total_time / 1000000.0f);

  // Always print class 14 (Stop sign) accuracy for Python to parse
  if (class_total[14] > 0) {
    float stop14_acc = (float)class_correct[14] / class_total[14] * 100.0f;
    printf("STOP14: %d/%d (%.1f%%)\n", class_correct[14], class_total[14], stop14_acc);
  } else {
    printf("STOP14: 0/0 (N/A)\n");
  }

  if (verbose_mode) {
    printf("\n==============================================\n");
    printf("  PER-CLASS ACCURACY\n");
    printf("==============================================\n");
    for (int c = 0; c < kCategoryCount; c++) {
      if (class_total[c] > 0) {
        float acc = (float)class_correct[c] / class_total[c] * 100.0f;
        printf("Class %2d: %2d/%2d (%5.1f%%) %s\n", c, class_correct[c],
               class_total[c], acc, kCategoryLabels[c]);
      }
    }
    printf("==============================================\n");
  }

  printf("TEST_COMPLETE\n");
}

int main() {
  stdio_init_all();
  sleep_ms(2000);

  printf("\n");
  printf("==============================================\n");
  printf("  MobileNet GTSRB Batch Accuracy Test (SWIFI)\n");
  printf("  Testing %d images\n", NUM_TEST_IMAGES);
  printf("==============================================\n\n");

  tflite::InitializeTarget();

  // Load model
  printf("Loading model...\n");
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: Model version mismatch\n");
    while (1)
      tight_loop_contents();
  }

  // Get total tensor count from the model flatbuffer
  total_tensors = (int)model->subgraphs()->Get(0)->tensors()->size();
  printf("Model has %d tensors\n", total_tensors);

  // Set up ops
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddQuantize();
  resolver.AddMean();
  resolver.AddPad();

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr,
      true /* preserve_all_tensors for SWIFI */);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors failed\n");
    while (1)
      tight_loop_contents();
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  printf("Arena used: %zu / %zu bytes\n", interpreter->arena_used_bytes(),
         (size_t)kTensorArenaSize);
  printf("Model ready!\n\n");

  char cmd[128];
  int tensor_idx = -1;
  int byte_offset = -1;
  int bit_in_byte = -1;

  // Auto-detect mode: wait 3 seconds for serial input.
  // If a host sends a command → interactive (SWIFI) mode.
  // If no serial arrives → continuous baseline mode (for GDB/OpenOCD).
  printf("READY_FOR_CMD\n");
  printf("[INFO] Waiting 3s for host connection...\n");

  bool host_connected = false;
  absolute_time_t deadline = make_timeout_time_ms(3000);
  while (!time_reached(deadline)) {
    int c = getchar_timeout_us(10000);  // 10ms poll
    if (c != PICO_ERROR_TIMEOUT) {
      host_connected = true;
      // Consume any remaining chars from this initial message
      while (getchar_timeout_us(50000) != PICO_ERROR_TIMEOUT) {}
      break;
    }
  }

  if (!host_connected) {
    // ─── AUTO-BASELINE MODE (for GDB/OpenOCD fault injection) ───
    printf("\n");
    printf("==============================================\n");
    printf("  AUTO-BASELINE MODE\n");
    printf("  No host detected. Running inference loop.\n");
    printf("  Use GDB to halt, inject faults, and resume.\n");
    printf("==============================================\n\n");

    int run_count = 0;
    while (true) {
      run_count++;

      // Check GDB fault slots
      if (gdb_fault_tensor >= 0 && gdb_fault_tensor < total_tensors) {
        // Restore any previous fault
        if (faulted_tensor_idx != -1) {
          TfLiteEvalTensor *t = interpreter->GetTensor(faulted_tensor_idx);
          t->data.data = original_data_ptr;
          faulted_tensor_idx = -1;
          original_data_ptr = nullptr;
        }

        int ft = gdb_fault_tensor;
        int fb = gdb_fault_byte;
        int fbit = gdb_fault_bit;

        TfLiteEvalTensor *t = interpreter->GetTensor(ft);
        size_t tensor_bytes = tflite::EvalTensorBytes(t);
        if (tensor_bytes <= sizeof(fault_buffer)) {
          original_data_ptr = t->data.data;
          memcpy(fault_buffer, original_data_ptr, tensor_bytes);
          if (fb >= 0 && (size_t)fb < tensor_bytes) {
            fault_buffer[fb] ^= (1 << fbit);
          }
          t->data.data = fault_buffer;
          faulted_tensor_idx = ft;
          printf("--- AUTO RUN %d [GDB FAULT: tensor=%d byte=%d bit=%d] ---\n",
                 run_count, ft, fb, fbit);
        } else {
          printf("--- AUTO RUN %d [GDB FAULT ERROR: tensor %d too large] ---\n",
                 run_count, ft);
        }

        // Clear the fault slot so next run is clean (unless GDB sets it again)
        gdb_fault_tensor = -1;
      } else {
        // Restore any previous fault for clean run
        if (faulted_tensor_idx != -1) {
          TfLiteEvalTensor *t = interpreter->GetTensor(faulted_tensor_idx);
          t->data.data = original_data_ptr;
          faulted_tensor_idx = -1;
          original_data_ptr = nullptr;
        }
        printf("--- AUTO RUN %d [CLEAN] ---\n", run_count);
      }

      // Brief pause — GDB can halt here to set gdb_fault_* before next run
      sleep_ms(100);

      run_batch_test();
    }
  }

  printf("[INFO] Host connected. Interactive mode.\n");

  while (true) {
    printf("READY_FOR_CMD\n");

    // Wait for command from host
    // Supported commands:
    //   INJECT:<tensor>,<byte_offset>,<bit>  - Inject fault and run batch
    //   BASELINE                             - Run batch without fault (clean model)
    //   QUERY:<tensor>                       - Report tensor size in bytes
    //   MODE:VERBOSE                         - Enable per-image output
    //   MODE:QUIET                           - Disable per-image output (fast)
    int cmd_pos = 0;
    bool cmd_received = false;
    while (!cmd_received) {
      int c = getchar_timeout_us(1000);
      if (c == PICO_ERROR_TIMEOUT)
        continue;

      if (c == '\n' || c == '\r') {
        if (cmd_pos > 0) {
          cmd[cmd_pos] = '\0';
          cmd_received = true;
          cmd_pos = 0;
        }
      } else if (cmd_pos < (int)sizeof(cmd) - 1) {
        cmd[cmd_pos++] = (char)c;
      }
    }

    // --- Handle QUERY command ---
    int query_tensor = -1;
    if (sscanf(cmd, "QUERY:%d", &query_tensor) == 1) {
      if (query_tensor >= 0 && query_tensor < total_tensors) {
        TfLiteEvalTensor *t = interpreter->GetTensor(query_tensor);
        size_t tensor_bytes = tflite::EvalTensorBytes(t);
        printf("SIZE:%d,%zu\n", query_tensor, tensor_bytes);
      } else {
        printf("ERROR: Tensor %d out of range (0-%d)\n", query_tensor,
               total_tensors - 1);
      }
      continue;
    }

    // --- Handle MODE command ---
    if (strncmp(cmd, "MODE:VERBOSE", 12) == 0) {
      verbose_mode = true;
      printf("MODE:VERBOSE\n");
      continue;
    }
    if (strncmp(cmd, "MODE:QUIET", 10) == 0) {
      verbose_mode = false;
      printf("MODE:QUIET\n");
      continue;
    }

    // --- Handle BASELINE command (run batch without fault) ---
    if (strncmp(cmd, "BASELINE", 8) == 0) {
      // Restore any previous fault first
      if (faulted_tensor_idx != -1) {
        TfLiteEvalTensor *t = interpreter->GetTensor(faulted_tensor_idx);
        t->data.data = original_data_ptr;
        faulted_tensor_idx = -1;
        original_data_ptr = nullptr;
      }
      printf("BASELINE_RUN\n");
      tensor_idx = -1;
      run_batch_test();
      continue;
    }

    // --- Handle INJECT command ---
    if (sscanf(cmd, "INJECT:%d,%d,%d", &tensor_idx, &byte_offset,
               &bit_in_byte) != 3) {
      printf("ERROR: Unknown command: %s\n", cmd);
      continue;
    }

    // Restore previous tensor pointer if it was deflected
    if (faulted_tensor_idx != -1) {
      TfLiteEvalTensor *t = interpreter->GetTensor(faulted_tensor_idx);
      t->data.data = original_data_ptr;
      faulted_tensor_idx = -1;
      original_data_ptr = nullptr;
    }

    // Inject new fault using pointer swapping
    if (tensor_idx >= 0) {
      TfLiteEvalTensor *t = interpreter->GetTensor(tensor_idx);
      size_t tensor_bytes = tflite::EvalTensorBytes(t);
      if (tensor_bytes <= sizeof(fault_buffer)) {
        // Save original pointer and copy content
        original_data_ptr = t->data.data;
        memcpy(fault_buffer, original_data_ptr, tensor_bytes);

        // Emulate fault if offset applies
        if (byte_offset >= 0 && (size_t)byte_offset < tensor_bytes) {
          fault_buffer[byte_offset] ^= (1 << bit_in_byte);
        }

        // Redirect pointer
        t->data.data = fault_buffer;
        faulted_tensor_idx = tensor_idx;
        printf("FAULT_INJECTED:%d,%d,%d\n", tensor_idx, byte_offset,
               bit_in_byte);
      } else {
        printf("ERROR: Tensor %d size %zu exceeds buffer %zu\n", tensor_idx,
               tensor_bytes, sizeof(fault_buffer));
        continue;
      }
    } else {
      printf("CLEAN_RUN\n");
    }

    run_batch_test();
  }

  return 0;
}
