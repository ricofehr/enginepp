/**
 *  @file front_wall.h
 *  @brief FrontWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_FRONTWALL_H_
#define NEXTFLOOR_PLAYGROUND_FRONTWALL_H_

#include "nextfloor/playground/width_wall.h"

#include <memory>
#include <vector>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace playground {

/**
 *  @class FrontWall
 *  @brief FrontWall : define front wall side in a Room
 */
class FrontWall : public WidthWall {

public:
    FrontWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    ~FrontWall() final = default;

    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_FRONTWALL_H_
