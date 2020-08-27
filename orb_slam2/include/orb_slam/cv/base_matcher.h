#pragma once

namespace orbslam {

class BaseMatcher {

public:
  BaseMatcher(/* args */);

  ~BaseMatcher();

  static void createMatcher() {}

  // virtual int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints,
  //                        const float th = 3) = 0;

private:
  /* data */
};

} // namespace orbslam
