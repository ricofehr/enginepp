/**
 *  @file game_camera_factory.h
 *  @brief GameCameraFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CAMERA_GAMECAMERAFACTORY_H_
#define NEXTFLOOR_CAMERA_GAMECAMERAFACTORY_H_

#include "nextfloor/character/camera_factory.h"

namespace nextfloor {

namespace camera {

/**
 *  @class GameCameraFactory
 *  @brief Concrete Factory for camera related objects
 */
class GameCameraFactory : public nextfloor::character::CameraFactory {

public:
    std::unique_ptr<nextfloor::character::Camera> MakeCamera() const override;
};

}  // namespace camera

}  // namespace nextfloor

#endif  // NEXTFLOOR_CAMERA_GAMECAMERAFACTORY_H_
