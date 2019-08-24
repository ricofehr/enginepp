/**
 *  @file back_wall.h
 *  @brief BackWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_BACKWALL_H_
#define NEXTFLOOR_OBJECTS_BACKWALL_H_

#include "nextfloor/objects/width_wall.h"

#include <glm/glm.hpp>
#include <string>


namespace nextfloor {

namespace objects {

/**
 *  @class BackWall
 *  @brief Back Wall Side in a Room
 */
class BackWall : public WidthWall {

public:
    BackWall(glm::vec3 location, glm::vec3 scale);

    BackWall(BackWall&&) = default;
    BackWall& operator=(BackWall&&) = default;
    BackWall(const BackWall&) = delete;
    BackWall& operator=(const BackWall&) = delete;

    virtual ~BackWall() override = default;

    virtual void PrepareDraw() noexcept override;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_BACKWALL_H_
