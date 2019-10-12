/**
 *  @file front_wall.h
 *  @brief FrontWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_FRONTWALL_H_
#define NEXTFLOOR_OBJECTS_FRONTWALL_H_

#include "nextfloor/objects/width_wall.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class FrontWall
 *  @brief FrontWall : define front wall side in a Room
 */
class FrontWall : public WidthWall {

public:
    FrontWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks);
    ~FrontWall() final = default;

    void PrepareDraw(const Camera& active_camera) final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_FRONTWALL_H_
