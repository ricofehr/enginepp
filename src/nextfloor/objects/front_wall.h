/**
 *  @file front_wall.h
 *  @brief FrontWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_FRONTWALL_H_
#define NEXTFLOOR_OBJECTS_FRONTWALL_H_

#include "nextfloor/objects/width_wall.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class FrontWall
 *  @brief FrontWall : define front wall side in a Room
 */
class FrontWall : public WidthWall {

public:
    FrontWall(const glm::vec3& location, const glm::vec3& scale);
    ~FrontWall() final = default;

    void PrepareDraw(const Camera& active_camera) final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_FRONTWALL_H_
