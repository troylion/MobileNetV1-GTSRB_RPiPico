/* MobileNet GTSRB Model Settings
   Traffic Sign Recognition using MobileNet
*/

#ifndef MOBILENET_GTSRB_MODEL_SETTINGS_H_
#define MOBILENET_GTSRB_MODEL_SETTINGS_H_

// Image dimensions - GTSRB uses 32x32 RGB
constexpr int kNumCols = 32;
constexpr int kNumRows = 32;
constexpr int kNumChannels = 3;
constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;  // 3072 bytes

// Output classes - GTSRB has 43 traffic sign types
constexpr int kCategoryCount = 43;

// Traffic sign category labels
extern const char* kCategoryLabels[kCategoryCount];

#endif  // MOBILENET_GTSRB_MODEL_SETTINGS_H_
