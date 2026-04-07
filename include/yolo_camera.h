#pragma once

#include <format>
#include <stdexcept>
#include <string>

#include <opencv2/videoio.hpp>

#include <trtyolo.hpp>

struct Config {
    int camera_id;
    std::string calibration_result_path;
    std::string model_path;
    std::string infer_option_path;
    std::string labels_path;
};

struct CalibrationResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

struct InferOptionConfig {
    int device_id = 0;                       // CUDA 设备 ID
    bool enable_swap_rb = false;             // 是否启用 BGR->RGB 转换
    bool cuda_memory = false;                // 是否启用 CUDA 内存
    bool managed_memory = false;             // 是否启用统一内存
    bool enable_performance_report = false;  // 是否启用性能报告
    std::optional<std::pair<int, int>>
        input_dimensions;  // 输入尺寸，可选，格式为(高，宽)
};

class YoloCamera {
private:
    Config cfg;
    cv::VideoCapture cap;
    CalibrationResult calib_res;

    std::vector<std::string> labels;
    trtyolo::InferOption infer_option;
    trtyolo::DetectModel* model;

    void read_config(const std::string& config_file);
    void read_calibration_result(const std::string& calib_result_path);
    void read_infer_option(const std::string& option_file);
    void read_labels(const std::string& labels_path);
    void create_camera(int camera_id);
    void create_model(const std::string& model_path);

    void init(const std::string& config_file);

public:
    YoloCamera() = delete;
    YoloCamera(const std::string& config_file);
    ~YoloCamera() {
        delete model;
    }

    YoloCamera(const YoloCamera&) = delete;
    YoloCamera& operator=(const YoloCamera&) = delete;

    const CalibrationResult& get_calibration_result() const {
        return calib_res;
    }

    const cv::VideoCapture& get_camera() const {
        return cap;
    }
    cv::VideoCapture& get_camera() {
        return const_cast<cv::VideoCapture&>(
            static_cast<const YoloCamera*>(this)->get_camera());
    }

    cv::Mat read_frame();

    trtyolo::Image input_frame_to_image(const cv::Mat& frame) const;

    trtyolo::DetectRes detect(const cv::Mat& frame);

    cv::Mat visualize(const cv::Mat& frame,
                      const trtyolo::DetectRes& res) const;

    const trtyolo::InferOption& get_infer_option() const {
        return infer_option;
    }

    trtyolo::InferOption& get_infer_option() {
        return const_cast<trtyolo::InferOption&>(
            static_cast<const YoloCamera*>(this)->get_infer_option());
    }

    const trtyolo::DetectModel& get_model() const {
        return *model;
    }

    trtyolo::DetectModel& get_model() {
        return const_cast<trtyolo::DetectModel&>(
            static_cast<const YoloCamera*>(this)->get_model());
    }
};

struct FileNotFoundException : public std::runtime_error {
public:
    FileNotFoundException(const std::string& file_path)
        : runtime_error("File not found: " + file_path) {}
};

struct CameraOpenException : public std::runtime_error {
    CameraOpenException(int camera_id)
        : std::runtime_error(std::format(
              "The camera with id {} cannot be opened.", camera_id)) {}
};

struct FrameCaptureException : public std::runtime_error {
    FrameCaptureException()
        : std::runtime_error("The camera cannot capture frames.") {}
};
