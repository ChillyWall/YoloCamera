#include <opencv2/core/persistence.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <yolo_camera.h>

void YoloCamera::read_config(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(CONFIG_FILE);
    }

    cfg.camera_id = static_cast<int>(fs["camera_id"]);
    cfg.calibration_result_path =
        static_cast<std::string>(fs["calibration_result_path"]);
    cfg.model_path = static_cast<std::string>(fs["model_path"]);
    cfg.infer_option_path = static_cast<std::string>(fs["infer_option_path"]);
    cfg.labels_path = static_cast<std::string>(fs["labels_file"]);
}

void YoloCamera::read_calibration_result(const std::string& calib_result_path) {
    cv::FileStorage fs(calib_result_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(CONFIG_FILE);
    }
    fs["cameraMatrix"] >> calib_res.camera_matrix;
    fs["distCoeffs"] >> calib_res.dist_coeffs;
}

void YoloCamera::read_infer_option(const std::string& option_file) {
    cv::FileStorage fs(option_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(CONFIG_FILE);
    }
    cv::FileNode node;
    if ((node = fs["device_id"]).isInt()) {
        infer_option.setDeviceId((int) node);
    }
    if ((node = fs["cuda_memory"]).isInt()) {
        if ((int) node)
            infer_option.enableCudaMem();
    }
    if ((node = fs["managed_memory"]).isInt()) {
        if ((int) node)
            infer_option.enableManagedMemory();
    }
    if ((node = fs["enable_swap_rb"]).isInt()) {
        if ((int) node)
            infer_option.enableSwapRB();
    }
    if ((node = fs["enable_performance_report"]).isInt()) {
        if ((int) node)
            infer_option.enablePerformanceReport();
    }
    if ((node = fs["input_dimensions"]).isSeq()) {
        std::vector<int> dims;
        node >> dims;
        if (dims.size() == 2) {
            infer_option.setInputDimensions(dims[0], dims[1]);
        }
    }
}

void YoloCamera::read_labels(const std::string& labels_path) {
    cv::FileStorage fs(labels_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(CONFIG_FILE);
    }

    cv::FileNode node = fs["labels"];
    if (node.isSeq()) {
        labels.clear();
        labels.reserve(node.size());
        for (const auto& label : node) {
            labels.push_back((std::string) label);
        }
    }
}

void YoloCamera::create_camera(int camera_id) {
    cap = cv::VideoCapture(camera_id);
    if (!cap.isOpened()) {
        throw CameraOpenException(camera_id);
    }
}

void YoloCamera::create_model(const std::string& model_path) {
    model = new trtyolo::DetectModel(model_path, infer_option);
}

cv::Mat YoloCamera::read_frame() {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        throw FrameCaptureException();
    }

    cv::Mat undist_frame;
    cv::undistort(frame, undist_frame, calib_res.camera_matrix,
                  calib_res.dist_coeffs);

    return undist_frame;
}

trtyolo::Image YoloCamera::input_frame_to_image(const cv::Mat& frame) const {
    return trtyolo::Image(frame.data, frame.cols, frame.rows);
}

trtyolo::DetectRes YoloCamera::detect(const cv::Mat& frame) {
    trtyolo::Image img = input_frame_to_image(frame);
    return model->predict(img);
}

cv::Mat YoloCamera::visualize(const cv::Mat& frame,
                              const trtyolo::DetectRes& res) const {
    cv::Mat image = frame.clone();
    for (size_t i = 0; i < res.num; ++i) {
        const auto& box = res.boxes[i];
        int cls = res.classes[i];
        float score = res.scores[i];
        const auto& label = labels[cls];
        std::string label_text = label + " " + cv::format("%.3f", score);

        // 绘制矩形和标签
        int base_line;
        cv::Size label_size = cv::getTextSize(
            label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top),
                      cv::Point(box.right, box.bottom),
                      cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height),
                      cv::Point(box.left + label_size.width, box.top),
                      cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208),
                    1);
    }
    return image;
}

void YoloCamera::init(const std::string& config_file) {
    read_config(config_file);
    read_calibration_result(cfg.calibration_result_path);
    read_labels(cfg.labels_path);
    create_camera(cfg.camera_id);
    read_infer_option(cfg.infer_option_path);
    create_model(cfg.model_path);
}

YoloCamera::YoloCamera(const std::string& config_file) {
    init(config_file);
}
