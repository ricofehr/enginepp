/**
 *  @file left_wall.h
 *  @brief LeftWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_LEFTWALL_H_
#define NEXTFLOOR_OBJECTS_LEFTWALL_H_

#include "nextfloor/objects/depth_wall.h"

#include <glm/glm.hpp>
#include <string>


namespace nextfloor {

namespace objects {

/**
 *  @class LeftWall
 *  @brief LeftWall
 */
class LeftWall : public DepthWall {

public:

    LeftWall(glm::vec3 location, glm::vec3 scale);

    LeftWall(LeftWall&&) = default;
    LeftWall& operator=(LeftWall&&) = default;

    LeftWall(const LeftWall&) = delete;
    LeftWall& operator=(const LeftWall&) = delete;

    ~LeftWall() override = default;

    virtual void PrepareDraw() noexcept override;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_LEFTWALL_H_
