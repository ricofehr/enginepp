/**
 *  @file right_wall.h
 *  @brief RightWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_RIGHTWALL_H_
#define NEXTFLOOR_OBJECTS_RIGHTWALL_H_

#include "nextfloor/objects/depth_wall.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class RightWall
 *  @brief RightWall : define right wall side in a Room
 */
class RightWall : public DepthWall {

public:
    RightWall(const glm::vec3& location, const glm::vec3& scale, const MeshFactory& factory);
    ~RightWall() final = default;

    void PrepareDraw(const Camera& active_camera) final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_RIGHTWALL_H_
