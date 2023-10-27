// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the gaze_estimation_demo application
* \file gaze_estimation_demo/main.cpp
* \example gaze_estimation_demo/main.cpp
*/
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include "openvino/openvino.hpp"

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "face_inference_results.hpp"
#include "face_detector.hpp"
#include "base_estimator.hpp"
#include "head_pose_estimator.hpp"
#include "landmarks_estimator.hpp"
#include "eye_state_estimator.hpp"
#include "gaze_estimator.hpp"
#include "results_marker.hpp"
#include "utils.hpp"

#include "gaze_estimation_demo.hpp"

using namespace gaze_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validating input arguments
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m.empty())
        throw std::logic_error("Parameter -m is not set");
    if (FLAGS_m_fd.empty())
        throw std::logic_error("Parameter -m_fd is not set");
    if (FLAGS_m_hp.empty())
        throw std::logic_error("Parameter -m_hp is not set");
    if (FLAGS_m_lm.empty())
        throw std::logic_error("Parameter -m_lm is not set");
    if (FLAGS_m_es.empty())
        throw std::logic_error("Parameter -m_es is not set");

    return true;
}

int key_width = 105;
int key_height = 130;
int diff_width = 13;
int diff_height = 20;

void set_char(const std::vector<char>& arr, int start, char& ch, const cv::Point& point){
    for(char c : arr){
        int end = start + key_width;
        if(start <= point.x && point.x <= end){
            ch = c;
            return;
        }
        start = end + diff_width;
    }
    return;
}

char parse_char(const cv::Point& point){
    char ch = '=';
    if(0 <= point.y && point.y <= 130){
        std::vector<char> arr = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\b'} ;
        set_char(arr, 0, ch, point);
    }else if(150 <= point.y && point.y <= 280){
        std::vector<char> arr = {'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '\b'} ;
        set_char(arr, 0, ch, point);
    }else if(300 <= point.y && point.y <= 430){
        std::vector<char> arr = {'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', '\n', '\n'};
        set_char(arr, 70, ch, point);
    }else if(450 <= point.y && point.y <= 580){
        std::vector<char> arr = {'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.'};
        set_char(arr, 120, ch, point);
    }else if(237 <= point.x && point.x <= 930){
        ch = ' ';
    }
    return ch;
}

void display_buf(const std::vector<char>& buff){
    if(buff.size() > 0){
        std::string myStr(buff.begin(), buff.end());
        std::cout << "\x1B[2J\x1B[H";
        std::cout << "You typed: " << myStr << '\n';
    }
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // Parsing and validating of input arguments
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // Load OpenVINO runtime
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        // Set up face detector and estimators
        FaceDetector faceDetector(core, FLAGS_m_fd, FLAGS_d_fd, FLAGS_t, FLAGS_fd_reshape);
        HeadPoseEstimator headPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
        LandmarksEstimator landmarksEstimator(core, FLAGS_m_lm, FLAGS_d_lm);
        EyeStateEstimator eyeStateEstimator(core, FLAGS_m_es, FLAGS_d_es);
        GazeEstimator gazeEstimator(core, FLAGS_m, FLAGS_d);

        // Put pointers to all estimators in an array so that they could be processed uniformly in a loop
        BaseEstimator* estimators[] = {&headPoseEstimator, &landmarksEstimator, &eyeStateEstimator, &gazeEstimator};
        // Each element of the vector contains inference results on one face
        std::vector<FaceInferenceResults> inferenceResults;
        bool flipImage = true;
        ResultsMarker resultsMarker(false, false, false, true, true);
        int delay = 1;
        std::string windowName = "Gaze estimation demo";

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(
            FLAGS_i, FLAGS_loop, read_type::efficient, 0, std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        cv::Size graphSize{frame.cols / 4, 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);
        cv::Mat src = cv::imread("keyboard.jpg");
        if (src.empty()) {
            std::cerr << "Error: Couldn't load the image." << std::endl;
            return 1;
        }
        // cv::Size newSize(1280, 720); // Desired width and height
        cv::Mat canvas = src.clone();
        auto start = std::chrono::steady_clock::now();
        char ch = '_';
        char old_ch;
        cv::Point curr_point(-1, -1);
        const int TIMEOUT = 700;
        std::vector<char> buff;
        double alpha = 0.4; 
        double beta = 1.0 - alpha;
        cv::Mat dst;
        do {
            if (flipImage) {
                cv::flip(frame, frame, 1);
            }
            // Infer results
            auto inferenceResults = faceDetector.detect(frame);
            for (auto& inferenceResult : inferenceResults) {
                for (auto estimator : estimators) {
                    estimator->estimate(frame, inferenceResult);
                }
            }
            old_ch = ch;
            
            // Display the results
            for (auto const& inferenceResult : inferenceResults) {
                resultsMarker.custom_mark(frame, inferenceResult, curr_point);
            }
            
            ch = parse_char(curr_point);
            
            // if valid 
            if(ch != '=' && ch != old_ch){
                start = std::chrono::steady_clock::now();
            }else if(ch != '=' && ch == old_ch){
                auto stop = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                if (duration.count() > TIMEOUT) {
                    buff.push_back(ch);
                    display_buf(buff);
                    start = std::chrono::steady_clock::now();
                }
            }

            if(ch == '\b' && buff.size() > 0){
                buff.pop_back();
            }else if(ch == '\n' && buff.size() > 0){
                buff.clear();
            }
            presenter.drawGraphs(canvas);
            

            if (FLAGS_r) {
                for (auto& inferenceResult : inferenceResults) {
                    slog::debug << inferenceResult << slog::endl;
                }
            }
            cv::addWeighted(frame, alpha, canvas, beta, 0.0, dst);  
            videoWriter.write(dst);

            if (!FLAGS_no_show) {
                cv::imshow(windowName, dst);
                int key = cv::waitKey(delay);
                resultsMarker.toggle(key);

                // Press 'Esc' to quit, 'f' to flip the video horizontally
                if (key == 27)
                    break;
                if (key == 'f')
                    flipImage = !flipImage;
                else
                    presenter.handleKey(key);
            }
            startTime = std::chrono::steady_clock::now();
            frame = cap->read();
            canvas = src.clone();
            display_buf(buff);
            // std::cout << "CH:" << ch << std::endl;
        } while (frame.data);

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
