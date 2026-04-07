#include <opencv2/opencv.hpp>

#include <trtyolo.hpp>

#include <yolo_camera.h>

int main() {
    YoloCamera yolo_camera(CONFIG_FILE);

    while (1) {
        auto frame = yolo_camera.read_frame();
        auto res = yolo_camera.detect(frame);
        auto res_img = yolo_camera.visualize(frame, res);
        cv::imshow("YoloCamera", res_img);

        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}
