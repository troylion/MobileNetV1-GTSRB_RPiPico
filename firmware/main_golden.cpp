/* MobileNet GTSRB Golden Model - Clean Inference Only
   
   This is a clean inference binary with NO fault injection logic.
   It runs MobileNet GTSRB inference in a loop, producing baseline results.
   
   Designed for external debugger-based fault injection (OpenOCD/GDB):
   - OpenOCD can halt the Pico at any point
   - GDB can modify SRAM (tensor arena, weights, etc.)
   - Resume inference and observe the effect
   
   Single-core only (TF_LITE_PICO_MULTICORE is disabled in the library).
*/

#include "pico/stdlib.h"
#include <cstdio>
#include <cstring>

#include "model_data_unbiased.h"
#include "model_settings.h"
#include "test_images_unbiased.h"

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
}  // namespace

struct InferenceResult {
  int predicted_class;
  int8_t confidence;
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

  // Find max score
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

void run_batch_test() {
  int correct = 0;
  int total = 0;
  uint32_t total_time = 0;

  int class_correct[kCategoryCount] = {0};
  int class_total[kCategoryCount] = {0};

  printf("\n##############################################\n");
  printf("  STARTING BATCH\n");
  printf("##############################################\n");

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

    const char *status = is_correct ? "OK" : "WRONG";
    printf("[%3d/%d] %s GT:%2d Pred:%2d Conf:%4d (%3d%%) %s (%lu ms)\n",
           i + 1, NUM_TEST_IMAGES, image_filenames[i], gt_class, pred_class,
           raw_score, confidence_pct, status, elapsed / 1000);
  }

  // Summary
  printf("\n");
  printf("==============================================\n");
  printf("  OVERALL RESULTS\n");
  printf("==============================================\n");
  printf("Correct: %d / %d\n", correct, total);
  printf("Accuracy: %.1f%%\n", (float)correct / total * 100.0f);
  printf("Avg inference time: %.1f ms\n",
         (float)total_time / total / 1000.0f);
  printf("Total time: %.1f seconds\n", (float)total_time / 1000000.0f);

  // Stop sign accuracy (class 14) for easy parsing
  if (class_total[14] > 0) {
    float stop14_acc = (float)class_correct[14] / class_total[14] * 100.0f;
    printf("STOP14: %d/%d (%.1f%%)\n", class_correct[14], class_total[14],
           stop14_acc);
  } else {
    printf("STOP14: 0/0 (N/A)\n");
  }

  // Per-class breakdown
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

  printf("TEST_COMPLETE\n");
}

int main() {
  stdio_init_all();
  sleep_ms(2000);

  printf("\n");
  printf("==============================================\n");
  printf("  MobileNet GTSRB Golden Model (Single-Core)\n");
  printf("  Clean inference — no fault injection\n");
  printf("  Testing %d images\n", NUM_TEST_IMAGES);
  printf("==============================================\n\n");

  tflite::InitializeTarget();

  // Load model
  printf("Loading model...\n");
  model = tflite::GetModel(model_data_unbiased);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: Model version mismatch\n");
    while (1)
      tight_loop_contents();
  }

  int total_tensors = (int)model->subgraphs()->Get(0)->tensors()->size();
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
  // preserve_all_tensors = true so GDB can inspect intermediate tensors
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr,
      true /* preserve_all_tensors */);
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

  // Run inference batch exactly once
  printf("--- GOLDEN RUN 1 ---\n");
  run_batch_test();
  printf("--- END OF EXECUTION ---\n");

  // Idle endlessly so OpenOCD can still inspect state without the processor resetting
  while (true) {
    tight_loop_contents();
  }

  return 0;
}
