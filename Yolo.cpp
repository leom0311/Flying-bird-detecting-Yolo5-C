#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define CLS_BIRD "bird"

void load_net(cv::dnn::Net& net, bool cuda, char *model_file)
{
    auto result = cv::dnn::readNet(model_file);
    if (cuda) {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;
    
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 6;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

#include <Windows.h>
#include <iostream>

void usage() {
    std::cout << "It can be used to detect birds in the sky using SSD algorithm." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:";
    std::cout << std::endl;
    std::cout << "  --video file: Video file to be detected" << std::endl;
    std::cout << "  --model file: Trained model to be used for detection" << std::endl;
    std::cout << "  --cuda: Use GPU to detect objects (enable only when model is compiled as GPU.)" << std::endl;
    std::cout << "  --save-result file: File path to save detection results" << std::endl;
    std::cout << "  --silence: Don't play video" << std::endl;

    std::cout << std::endl;
    std::cout << "For example:" << std::endl;
    std::cout << "  bird-detector.exe --video \"6 birds agianst sky iphone 14 video.MOV\" --model v5.onnx --cuda --save-result \"result\\output.txt\" --silence";
}

void write2file(FILE *fp, cv::Rect box) {
    if (!fp) {
        return;
    }
    char buf[0x400] = { 0 };
    fwrite(buf, 1, strlen(buf), fp);
    sprintf_s(buf, "  {\"x\":%d,   \"y\":%d,    \"width:\":%d,   \"height\":%d}\n",
        box.x,
        box.y,
        box.width,
        box.height);
    fwrite(buf, 1, strlen(buf), fp);
}

typedef
struct dustMeta {
    bool is_dust;
    int confirm;
    unsigned long long last_scene;
    cv::Rect area;
} dustMeta;

int conflict(std::vector<dustMeta> dusts, cv::Rect rect) {
    for (int i = 0; i < dusts.size(); i++) {
        cv::Rect intersection = dusts[i].area & rect;
        if (intersection.area() > (rect.area() * 0.5)) {
            return i;
        } 
    }
    return -1;
}

cv::Scalar calculateAverageColor(const cv::Mat& image, const cv::Rect& rect) {
    // Ensure the rectangle is within the image boundaries
    cv::Rect boundedRect = rect & cv::Rect(0, 0, image.cols, image.rows);

    // Extract the ROI
    cv::Mat roi = image(boundedRect);

    // Calculate the mean color value within the ROI
    cv::Scalar averageColor = cv::mean(roi);

    return averageColor;
}

int main(int argc, char** argv)
{
    char video_file[MAX_PATH] = { 0 };
    char model_file[MAX_PATH] = { 0 };
    char result_file[MAX_PATH] = { 0 };
    bool cuda = false;
    bool silence = false;

    if (argc == 1) {
        usage();
        return -1;
    }
    for (auto i = 1; i < argc; i++) {
        if (!strncmp("/?", argv[i], strlen("/?"))) {
            usage();
            return -1;
        }
        else if (!strncmp("--video", argv[i], strlen("--video"))) {
            strcpy_s(video_file, argv[i + 1]);
            i++;
        }
        else if (!strncmp("--model", argv[i], strlen("--model"))) {
            strcpy_s(model_file, argv[i + 1]);
            i++;
        }
        else if (!strncmp("--save-result", argv[i], strlen("--save-result"))) {
            strcpy_s(result_file, argv[i + 1]);
            i++;
        }
        else if (!strncmp("--cuda", argv[i], strlen("--cuda"))) {
            cuda = true;
        }
        else if (!strncmp("--silence", argv[i], strlen("--silence"))) {
            silence = true;
        }
        else {
            usage();
            return -1;
        }
    }
    if (!video_file[0]) {
        std::cout << "Please input video file" << std::endl;
        usage();
        return -1;
    }
    if (!model_file[0]) {
        std::cout << "Please input model file" << std::endl;
        usage();
        return -1;
    }
    if (cuda) {
        std::cout << "Cuda is diabled" << std::endl;
    }
    if (silence) {
        std::cout << "Dont play video" << std::endl;
    }
    
    std::vector<std::string> class_list;
    class_list.push_back(CLS_BIRD);

    cv::Mat frame;
    cv::VideoCapture capture(video_file);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video file\n";
        return -1;
    }

    cv::dnn::Net net;
    load_net(net, cuda, model_file);

    int frame_offset = 0;
    FILE* fp = 0;
    fopen_s(&fp, "report.txt", "wb");
    if (!fp) {
        std::cout << "Failed to open file to save detection result" << std::endl;
        capture.release();
        return -1;
    }

    std::vector<dustMeta> dusts;
    while (true) {
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "Reached at EOF\n";
            break;
        }
        std::cout << "Processing " << frame_offset + 1 << "th frame" << std::endl;
        
        std::vector<Detection> output;
        detect(frame, net, output, class_list);
        int detections = output.size();
        if (detections > 0 && fp) {
            char buf[0x100];
            sprintf_s(buf, "frame: %d\n", frame_offset + 1);
            fwrite(buf, 1, strlen(buf), fp);
        }

        for (int i = 0; i < detections; ++i) {
            Detection detection = output[i];
            cv::Rect box = detection.box;

            write2file(fp, box);
            if (!silence) {
                const cv::Scalar color = cv::Scalar(0, 0, 255);
                cv::rectangle(frame, box, color, 1);
            }
        }
        // capture images to file -->
        if (detections) {
            char szCapture[260];
            sprintf_s(szCapture, "images/%d.jpg", frame_offset);
            cv::imwrite(szCapture, frame);
        }
        // <-- capture images to file

        if (!silence) {
            cv::imshow("output", frame);
            if (cv::waitKey(1) != -1) {
                std::cout << "Terminated by interrupt\n";
                break;
            }
        }
        frame_offset++;
    }
    if (fp) {
        fclose(fp);
    }
    capture.release();
    std::cout << "Total: " << frame_offset << "\n";
    return 0;
}
