/**
 *  @file left_wall.h
 *  @brief LeftWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_LEFTWALL_H_
#define NEXTFLOOR_PLAYGROUND_LEFTWALL_H_

#include "nextfloor/playground/depth_wall.h"

#include <memory>
#include <glm/glm.hpp>

namespace nextfloor {

namespace playground {

/**
 *  @class LeftWall
 *  @brief LeftWall : define left wall side in a Room
 */
class LeftWall : public DepthWall {

public:
    LeftWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    ~LeftWall() final = default;

    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_LEFTWALL_H_
