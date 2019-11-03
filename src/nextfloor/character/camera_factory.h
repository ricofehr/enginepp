/**
 *  @file camera_factory.h
 *  @brief CameraFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_CAMERAFACTORY_H_
#define NEXTFLOOR_CHARACTER_CAMERAFACTORY_H_

#include <memory>

#include "nextfloor/character/camera.h"

namespace nextfloor {

namespace character {

/**
 *  @class CameraFactory
 *  @brief Abstract Factory Pattern for Cameras
 */
class CameraFactory {

public:
    virtual ~CameraFactory() = default;
    virtual std::unique_ptr<Camera> MakeCamera() const = 0;
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_CAMERAFACTORY_H_
