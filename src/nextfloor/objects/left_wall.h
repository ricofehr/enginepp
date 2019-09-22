/**
 *  @file left_wall.h
 *  @brief LeftWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_LEFTWALL_H_
#define NEXTFLOOR_OBJECTS_LEFTWALL_H_

#include "nextfloor/objects/depth_wall.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class LeftWall
 *  @brief LeftWall : define left wall side in a Room
 */
class LeftWall : public DepthWall {

public:
    LeftWall(const glm::vec3& location, const glm::vec3& scale);
    ~LeftWall() final = default;

    void PrepareDraw() final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_LEFTWALL_H_
