/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <mutex>
#include <set>

#include "keyframe.h"
#include "mappoint.h"

namespace orbslam {

class MapPoint;
class KeyFrame;

class Map {
public:
  Map();

  void AddKeyFrame(KeyFrame *pKF);
  void AddMapPoint(MapPoint *pMP);
  void EraseMapPoint(MapPoint *pMP);
  void EraseKeyFrame(KeyFrame *pKF);
  void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

  std::mutex mutex_bakfs;
  void setBAKFs(const std::vector<KeyFrame *> &kfs,
                const std::vector<KeyFrame *> &kfs_fixed);
  std::vector<KeyFrame *> getBAKeyFrames();
  std::vector<KeyFrame *> getBAKeyFramesFixed();
  std::vector<KeyFrame *> mvpBAKFs, mvpBAKFsFixed;

  std::vector<KeyFrame *> GetAllKeyFrames();
  std::vector<MapPoint *> GetAllMapPoints();
  std::vector<MapPoint *> GetReferenceMapPoints();

  long unsigned int MapPointsInMap();
  long unsigned KeyFramesInMap();

  long unsigned int GetMaxKFid();

  void clear();

  // ADD(dust visualization)
  KeyFrame *pDustRef;

  std::mutex mutex_lastkf;
  KeyFrame *pLastKF = nullptr;

  std::vector<KeyFrame *> mvpKeyFrameOrigins;

  std::mutex mMutexMapUpdate;

  // This avoid that two points are created simultaneously in separate threads
  // (id conflict)
  std::mutex mMutexPointCreation;

protected:
  std::set<MapPoint *> mspMapPoints; ///< MapPoints
  std::set<KeyFrame *> mspKeyFrames; ///< Keyframs

  std::vector<MapPoint *> mvpReferenceMapPoints;

  long unsigned int mnMaxKFid;

  std::mutex mMutexMap;
};

} // namespace orbslam
