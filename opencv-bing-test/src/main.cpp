#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

bool myfunction (int i, int j) { return (i<j); }

int main (int argc, const char * argv[]) {



    Mat image = imread("data/dog.jpg");
    string training_path ="opencv-bing-test/saliency/samples/ObjectnessTrainedModel";
    imshow("original", image);

    Ptr<ObjectnessBING> objectnessBING = makePtr<ObjectnessBING>();
    objectnessBING->setTrainingPath(training_path);
    objectnessBING->setBBResDir(training_path + "/temp");

    int a = rand()*100;
    vector<Vec4i> objectnessBoundingBox;
    if (objectnessBING->computeSaliency(image, objectnessBoundingBox) ) {
        vector<float> values = objectnessBING->getobjectnessValues();

        printf("detected candidates: %d\n", objectnessBoundingBox.size());
        printf("scores: %d\n", values.size());

        // The result are sorted by objectness. We uonly use the first 20 boxes here.
        for (int i = 0; i < 20; i++) {
            Mat clone = image.clone();
            Vec4i bb = objectnessBoundingBox[i];
            printf("index=%d, value=%f\n", i, values[i]);
            rectangle(clone, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(0, 0, 255), 4);

            char label[256];
            sprintf(label, "#%d", i+1);
            putText(clone, label, Point(bb[0], bb[1]+30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);

            char filename[256];
            sprintf(filename, "bing_%05d.jpg", i);
            imwrite(filename, clone);
        }
    }
    return 0;
}