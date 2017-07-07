/*!
 *  Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

//
//  File: image-classification-predict.cpp
//  This is a simple predictor which shows
//  how to use c api for image classfication
//  It uses opencv for image reading
//  Created by liuxiao on 12/9/15.
//  Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
//  Home Page: www.liuxiao.org
//  E-mail: liuxiao@foxmail.com
//

#include <stdio.h>

// Path for c_predict_api
#include <c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


// Read file to buffer
class BufferFile {
public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
            :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
            delete[] buffer_;
            buffer_ = NULL;
        }
    }
};

void GetImageFile(const std::string image_file,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b=123.68;
    float mean_g=116.78;
    float mean_r=103.94;
    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
                mean_data++;
            }
            if (channels > 1) {
                *ptr_image_g++ = mx_float(0.017)*(static_cast<mx_float>(*data++) - mean_g);
                *ptr_image_b++ = mx_float(0.017)*(static_cast<mx_float>(*data++) - mean_b);
            }

            *ptr_image_r++ = mx_float(0.017)*(static_cast<mx_float>(*data++) - mean_r);
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(std::string synset_file) {
    std::ifstream fi(synset_file.c_str());

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
           synset[best_idx].c_str(), best_idx, best_accuracy);
}

int main(int argc, char* argv[]) {


    std::string test_file = "data/2.jpg";

    // Models path for your model, you have to modify it
    std::string json_file = "mobilenet/mobilenet-symbol.json";
    std::string param_file = "mobilenet/mobilenet-0000.params";
    std::string synset_file = "mobilenet/synset.txt";


    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char *input_key[1] = {"data"};
    const char **input_keys = input_key;

    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = {0, 4};
    const mx_uint input_shape_data[4] = {1,
                                         static_cast<mx_uint>(channels),
                                         static_cast<mx_uint>(height),
                                         static_cast<mx_uint>(width)};
    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

    // Create Predictor
    MXPredCreate((const char *) json_data.GetBuffer(),
                 (const char *) param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    int image_size = width * height * channels;



    // Read Image Data
    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

    GetImageFile(test_file, image_data.data(), channels, cv::Size(width, height));

    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), image_size);

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);



    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset(synset_file);

    // Print Output Data
   PrintOutputResult(data, synset);

    return 0;
}
