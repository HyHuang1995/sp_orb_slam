#include "orb_slam/cv/base_extractor.h"

using namespace cv;

const int EDGE_THRESHOLD = 19;

namespace orbslam {

void BaseExtractor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];
    Size sz(cvRound((float)image.cols * scale),
            cvRound((float)image.rows * scale));
    Size wholeSize(sz.width + EDGE_THRESHOLD * 2,
                   sz.height + EDGE_THRESHOLD * 2);
    Mat temp(wholeSize, image.type()), masktemp;
    mvImagePyramid[level] =
        temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

    // Compute the resized image
    if (level != 0) {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0,
             cv::INTER_LINEAR);

      copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD,
                     EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     BORDER_REFLECT_101 + BORDER_ISOLATED);
    } else {
      copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101);
    }
  }
}
} // namespace orbslam