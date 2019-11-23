/**
 *  @file width_wall.h
 *  @brief WidthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_WIDTHWALL_H_
#define NEXTFLOOR_PLAYGROUND_WIDTHWALL_H_

#include "nextfloor/playground/wall.h"

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace playground {

/**
 *  @class WidthWall
 *  @brief WidthWall is an Abstract class for front / back walls of a Room
 */
class WidthWall : public Wall {

public:
    static constexpr char kTEXTURE[] = "assets/wall.png";

    static constexpr float kBRICK_WIDTH = 2.0f;
    static constexpr float kBRICK_HEIGHT = 2.0f;
    static constexpr float kBRICK_DEPTH = 0.25f;

    ~WidthWall() override = default;

    void AddDoor() final;
    void AddWindow() final;

protected:
    WidthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
