/**
 *
 *
 *
 * this file is just simple detector using YoloV3 with mobilenet for detection
 *
 *
 */
#include <caffe/caffe.hpp>
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/util/benchmark.hpp"
#include "thor/os.h"
#include "thor/vis.h"

/*char* CLASSES[81] = { "__background__",
"person", "bicycle", "car", "motorcycle",
"airplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter",
"bench", "bird", "cat",
"dog", "horse", "sheep", "cow",
"elephant", "bear", "zebra", "giraffe" ,
"backpack", "umbrella", "handbag", "tie" ,
"suitcase", "frisbee", "skis", "snowboard" ,
"sports ball", "kite", "baseball bat", "baseball glove" ,
"skateboard", "surfboard", "tennis racket", "bottle" ,
"wine glass", "cup", "fork", "knife" ,
"spoon", "bowl", "banana", "apple" ,
"sandwich", "orange", "broccoli", "carrot" ,
"hot dog", "pizza", "donut", "cake" ,
"chair", "couch", "potted plant", "bed" ,
"dining table", "toilet", "tv", "laptop" ,
"mouse", "remote", "keyboard", "cell phone" ,
"microwave", "oven", "toaster", "sink" ,
"refrigerator", "book", "clock", "vase" ,
"scissors", "teddy bear", "hair drier", "toothbrush" ,
};*/

vector<string> CLASSES = {"__background__",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor"};


using namespace caffe;  // NOLINT(build/namespaces)


class Detector {
public:
    Detector(const string &model_file,
             const string &weights_file,
             const string &mean_file,
             const string &mean_value,
             const float confidence_threshold,
             const float normalize_value,
             const string &cpu_mode,
             const int resize);

    std::vector <vector<float>> Detect(cv::Mat &img);

private:
    void setMean(const string &mean_file, const string &mean_value);

    void WrapInputLayer(std::vector <cv::Mat> *input_channels);

    cv::Mat preprocess(const cv::Mat &img,
                       std::vector <cv::Mat> *input_channels);

    cv::Mat preprocess(const cv::Mat &img,
                       std::vector <cv::Mat> *input_channels, double normalize_value);

    cv::Mat LetterBoxResize(cv::Mat img, int w, int h);

private:
    boost::shared_ptr <Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    float nor_val = 1.0;
    int resize_mode = 0;
public:
    float w_scale = 1;
    float h_scale = 1;
};


Detector::Detector(const string &model_file,
                   const string &weights_file,
                   const string &mean_file,
                   const string &mean_value,
                   const float confidence_threshold,
                   const float normalize_value,
                   const string &cpu_mode,
                   const int resize) {
  if (cpu_mode == "cpu") {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
  }
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
          << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  setMean(mean_file, mean_value);
  nor_val = normalize_value;
  resize_mode = resize;
}

float sec(clock_t clocks) {
  return (float) clocks / CLOCKS_PER_SEC;
}

std::vector <vector<float>> Detector::Detect(cv::Mat &img) {
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector <cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  if (nor_val != 1.0) {
    img = preprocess(img, &input_channels, nor_val);
  } else {
    img = preprocess(img, &input_channels);
  }
  clock_t time;
  time = clock();
  net_->Forward();
  printf("Predicted in %f seconds.\n", sec(clock() - time));
  /* Copy the output layer to a std::vector */
  Blob<float> *result_blob = net_->output_blobs()[0];
  const float *result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector <vector<float>> detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::setMean(const string &mean_file, const string &mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
                              "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector <cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
                             "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
                                                                "Specify either 1 mean_value or as many as channels: "
                                                                << num_channels_;

    std::vector <cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector <cv::Mat> *input_channels) {
  Blob<float> *input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

cv::Mat Detector::LetterBoxResize(cv::Mat img, int w, int h) {
  cv::Mat intermediateImg, outputImg;
  int delta_w, delta_h, top, left, bottom, right;
  int new_w = img.size().width;
  int new_h = img.size().height;

  if (((float) w / img.size().width) < ((float) h / img.size().height)) {
    new_w = w;
    new_h = (img.size().height * w) / img.size().width;
  } else {
    new_h = h;
    new_w = (img.size().width * h) / img.size().height;
  }
  cv::resize(img, intermediateImg, cv::Size(new_w, new_h));
  w_scale = w / (float) new_w;
  h_scale = h / (float) new_h;
  delta_w = w - new_w;
  delta_h = h - new_h;
  top = floor(delta_h / 2);
  bottom = delta_h - floor(delta_h / 2);
  left = floor(delta_w / 2);
  right = delta_w - floor(delta_w / 2);
  cv::copyMakeBorder(intermediateImg, outputImg, top, bottom, left, right, cv::BORDER_CONSTANT, (0, 0, 0));

  return outputImg;
}

cv::Mat Detector::preprocess(const cv::Mat &img,
                             std::vector <cv::Mat> *input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample, resized_img;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_) {
    if (resize_mode == 1) {
      sample_resized = LetterBoxResize(sample, input_geometry_.width, input_geometry_.height);
      resized_img = sample_resized;
    } else {
      cv::resize(sample, sample_resized, input_geometry_);
    }

  } else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
  * input layer of the network because it is wrapped by the cv::Mat
  * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
  if (resize_mode == 1)
    return resized_img;
  else
    return img;
}

cv::Mat Detector::preprocess(const cv::Mat &img,
                             std::vector <cv::Mat> *input_channels, double normalize_value) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample, resized_img;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_) {
    if (resize_mode == 1) {
      sample_resized = LetterBoxResize(sample, input_geometry_.width, input_geometry_.height);
      resized_img = sample_resized;
    } else {
      cv::resize(sample, sample_resized, input_geometry_);
    }
  } else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
  else
    sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
  if (resize_mode == 1)
    return resized_img;
  else
    return img;
}

DEFINE_string(mean_file, "", "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123", "If specified, can be one value or can be same as image channels Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file, "path/to/image_list/or/video",  "The file to images dir or video path, if none, inference on webcam.");
DEFINE_double(confidence_threshold, 0.90, "Only store detections with score higher than the threshold.");
DEFINE_double(normalize_value, 1.0,  "Normalize image to 0~1");
DEFINE_int32(resize_mode, 0, "0:WARP , 1:FIT_LARGE_SIZE_AND_PAD");
DEFINE_string(cpu_mode, "cpu", "cpu , gpu");


int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 0;
  gflags::SetUsageMessage("Do detection using YoloV3 MobileNet.\n"
                          "Usage:\n"
                          "    yolo_det [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/yolo/yolo_det");
    return 1;
  }

  const string &model_file = argv[1];
  const string &weights_file = argv[2];
  const string &mean_file = FLAGS_mean_file;
  const string &mean_value = FLAGS_mean_value;
  const string &file = FLAGS_file;
  const float &confidence_threshold = FLAGS_confidence_threshold;
  const float &normalize_value = FLAGS_normalize_value;
  const int &resize_mode = FLAGS_resize_mode;
  const string &cpu_mode = FLAGS_cpu_mode;
  // Initialize the network.

  Detector detector(model_file, weights_file, mean_file, mean_value, confidence_threshold, normalize_value, cpu_mode,
                    resize_mode);

  // for visualize
  const int font = cv::FONT_HERSHEY_COMPLEX_SMALL;
  const float font_scale = 0.4;
  const int font_thickness = 1;

  cout << "~~~~~~~~~ Start show demo ~~~~~~~~~~~~~~\n";
  if (thor::os::suffix(file) == "mp4" || thor::os::suffix(file) == "avi") {
    // inference on video
    cout << "inference on video " << file << endl;
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
      LOG(FATAL) << "Failed to open video: " << file;
    }
    cv::Mat img;
    int frame_count = 0;
    while (true) {
      bool success = cap.read(img);
      CHECK(!img.empty()) << "Error when read frame";
      std::vector <vector<float>> detections = detector.Detect(img);
      if (!success) { cap.release(); return 0;}

      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float> &d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          cv::Point pt1, pt2;
          pt1.x = (img.cols * d[3]);
          pt1.y = (img.rows * d[4]);
          pt2.x = (img.cols * d[5]);
          pt2.y = (img.rows * d[6]);

          thor::vis::RGBA unique_color_cao = thor::vis::gen_unique_color(d[1], 1.0 / CLASSES.size(), 0.7);
          cv::Scalar unique_color(unique_color_cao.r, unique_color_cao.g, unique_color_cao.b, unique_color_cao.a);
          cv::rectangle(img, pt1, pt2, unique_color, 1, 8, 0);

          char score_str[256];
          sprintf(score_str, "%.2f", score);
          std::string label_text = CLASSES[d[1]] + " " + string(score_str);
          int base_line = 0;
          cv::Point text_origin = cv::Point(pt1.x, pt1.y - 5);
          cv::Size text_size = cv::getTextSize(label_text, font, font_scale, font_thickness, &base_line);
          cv::rectangle(img, cv::Point(text_origin.x, text_origin.y + 5),
                        cv::Point(text_origin.x + text_size.width, text_origin.y - text_size.height - 5),
                        unique_color, -1, 0);
          cv::putText(img, label_text, text_origin, font, font_scale, cv::Scalar(255, 255, 255), font_thickness);
        }
      }
      cv::imshow("show", img);
      cv::waitKey(1);
      ++frame_count;
    }
  } else if (thor::os::isdir(file)){
    // inference on image list
    LOG(INFO) << "inference on images list\n";
    vector <cv::String> fn;
    cv::glob(file, fn, true); // recurse
    for (size_t k = 0; k < fn.size(); ++k) {
      cv::Mat img = cv::imread(fn[k]);
      if (img.empty()) continue; //only proceed if sucsessful
      CHECK(!img.empty()) << "Unable to decode image " << file;
      CPUTimer batch_timer;
      batch_timer.Start();
      std::vector <vector<float>> detections = detector.Detect(img);
      LOG(INFO) << "Computing time: " << batch_timer.MilliSeconds() << " ms.";
      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float> &d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          cv::Point pt1, pt2;
          pt1.x = (img.cols * d[3]);
          pt1.y = (img.rows * d[4]);
          pt2.x = (img.cols * d[5]);
          pt2.y = (img.rows * d[6]);

          thor::vis::RGBA unique_color_cao = thor::vis::gen_unique_color(d[1], 1.0 / CLASSES.size(), 0.7);
          cv::Scalar unique_color(unique_color_cao.r, unique_color_cao.g, unique_color_cao.b, unique_color_cao.a);
          cv::rectangle(img, pt1, pt2, unique_color, 1, 8, 0);

          char score_str[256];
          sprintf(score_str, "%.2f", score);
          std::string label_text = CLASSES[d[1]] + " " + string(score_str);
          int base_line = 0;
          cv::Point text_origin = cv::Point(pt1.x, pt1.y - 5);
          cv::Size text_size = cv::getTextSize(label_text, font, font_scale, font_thickness, &base_line);
          cv::rectangle(img, cv::Point(text_origin.x, text_origin.y + 5),
                        cv::Point(text_origin.x + text_size.width, text_origin.y - text_size.height - 5),
                        unique_color, -1, 0);
          cv::putText(img, label_text, text_origin, font, font_scale, cv::Scalar(255, 255, 255), font_thickness);
        }
      }
      cv::imshow("show", img);
      cv::imwrite("result_" + to_string(k) + ".jpg", img);
      cv::waitKey(0);
    }

  } else {
    LOG(INFO) << "inference on webcam\n";
    cout << "to be done.\n";

  }
}


