/**
 *  @file game_camera_factory.cc
 *  @brief GameCameraFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/camera/game_camera_factory.h"

#include "nextfloor/camera/head_camera.h"

namespace nextfloor {

namespace camera {

std::unique_ptr<nextfloor::character::Camera> GameCameraFactory::MakeCamera() const
{
    return std::make_unique<HeadCamera>();
}

}  // namespace camera

}  // namespace nextfloor