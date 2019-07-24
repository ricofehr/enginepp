/**
 *  @file right_wall.h
 *  @brief RightWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_RIGHTWALL_H_
#define NEXTFLOOR_OBJECTS_RIGHTWALL_H_

#include "nextfloor/objects/depth_wall.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class RightWall
 *  @brief RightWall : define right wall side in a Room
 */
class RightWall : public DepthWall {

public:

    RightWall(glm::vec3 location, glm::vec3 scale);
    RightWall(RightWall&&) = default;
    RightWall& operator=(RightWall&&) = default;
    RightWall(const RightWall&) = delete;
    RightWall& operator=(const RightWall&) = delete;
    ~RightWall() override = default;

    virtual void PrepareDraw() noexcept override;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_RIGHTWALL_H_
