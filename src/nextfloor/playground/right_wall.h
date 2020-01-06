/**
 *  @file right_wall.h
 *  @brief RightWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_RIGHTWALL_H_
#define NEXTFLOOR_PLAYGROUND_RIGHTWALL_H_

#include "nextfloor/playground/depth_wall.h"

#include <memory>
#include <vector>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace playground {

/**
 *  @class RightWall
 *  @brief RightWall : define right wall side in a Room
 */
class RightWall : public DepthWall {

public:
    RightWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    ~RightWall() final = default;

    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_RIGHTWALL_H_
