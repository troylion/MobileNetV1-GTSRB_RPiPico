/* MobileNet GTSRB Model Settings Implementation */

#include "model_settings.h"

// GTSRB Traffic Sign Category Labels
const char* kCategoryLabels[kCategoryCount] = {
    "Speed limit (20km/h)",              // 0
    "Speed limit (30km/h)",              // 1
    "Speed limit (50km/h)",              // 2
    "Speed limit (60km/h)",              // 3
    "Speed limit (70km/h)",              // 4
    "Speed limit (80km/h)",              // 5
    "End of speed limit (80km/h)",       // 6
    "Speed limit (100km/h)",             // 7
    "Speed limit (120km/h)",             // 8
    "No passing",                        // 9
    "No passing for large vehicles",     // 10
    "Right-of-way at intersection",      // 11
    "Priority road",                     // 12
    "Yield",                             // 13
    "Stop",                              // 14
    "No vehicles",                       // 15
    "Trucks prohibited",                 // 16
    "No entry",                          // 17
    "General caution",                   // 18
    "Dangerous curve left",              // 19
    "Dangerous curve right",             // 20
    "Double curve",                      // 21
    "Bumpy road",                        // 22
    "Slippery road",                     // 23
    "Road narrows on right",             // 24
    "Road work",                         // 25
    "Traffic signals",                   // 26
    "Pedestrians",                       // 27
    "Children crossing",                 // 28
    "Bicycles crossing",                 // 29
    "Beware of ice/snow",                // 30
    "Wild animals crossing",             // 31
    "End of all limits",                 // 32
    "Turn right ahead",                  // 33
    "Turn left ahead",                   // 34
    "Ahead only",                        // 35
    "Go straight or right",              // 36
    "Go straight or left",               // 37
    "Keep right",                        // 38
    "Keep left",                         // 39
    "Roundabout mandatory",              // 40
    "End of no passing",                 // 41
    "End of no passing by trucks"        // 42
};
