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

    FrontWall(glm::vec3 location, glm::vec3 scale);

    FrontWall(FrontWall&&) = default;
    FrontWall& operator=(FrontWall&&) = default;
    FrontWall(const FrontWall&) = delete;
    FrontWall& operator=(const FrontWall&) = delete;

    virtual ~FrontWall() override = default;

    virtual void PrepareDraw() noexcept override;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_FRONTWALL_H_
