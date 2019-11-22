/**
 *  @file depth_wall.h
 *  @brief DepthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_DEPTHWALL_H_
#define NEXTFLOOR_PLAYGROUND_DEPTHWALL_H_

#include "nextfloor/playground/wall.h"

#include <memory>
#include <glm/glm.hpp>
#include <string>

namespace nextfloor {

namespace playground {

/**
 *  @class DepthWall
 *  @brief DepthWall is an Abstract class for left / right wall of a Room
 */
class DepthWall : public Wall {

public:
    static constexpr char kTEXTURE[] = "assets/wall.png";

    static constexpr float kBRICK_WIDTH = 0.25;
    static constexpr float kBRICK_HEIGHT = 2.0f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    ~DepthWall() override = default;

    void AddDoor() final;
    void AddWindow() final;

protected:
    DepthWall(std::unique_ptr<nextfloor::mesh::Border> border,
              std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_DEPTHWALL_H_
