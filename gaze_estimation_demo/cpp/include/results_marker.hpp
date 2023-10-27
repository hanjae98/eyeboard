// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include "face_inference_results.hpp"

namespace gaze_estimation {
class ResultsMarker {
public:
    ResultsMarker(bool showFaceBoundingBox,
                  bool showHeadPoseAxes,
                  bool showLandmarks,
                  bool showGaze,
                  bool showEyeState);
    void mark(cv::Mat& image, const FaceInferenceResults& faceInferenceResults) const;
    void mark2(cv::Mat& image, const FaceInferenceResults& faceInferenceResults, cv::Mat& canvas) const;
    void custom_mark(cv::Mat& image, const FaceInferenceResults& faceInferenceResults, cv::Point& curr_point) const;
    void toggle(int key);

private:
    bool showFaceBoundingBox;
    bool showHeadPoseAxes;
    bool showLandmarks;
    bool showGaze;
    bool showEyeState;
};
}  // namespace gaze_estimation
