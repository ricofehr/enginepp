/**
 *  @file width_wall.h
 *  @brief WidthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_WIDTHWALL_H_
#define NEXTFLOOR_PLAYGROUND_WIDTHWALL_H_

#include "nextfloor/playground/wall.h"

#include <memory>
#include <vector>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace playground {

/**
 *  @class WidthWall
 *  @brief WidthWall is an Abstract class for front / back walls of a Room
 */
class WidthWall : public Wall {

public:
    static constexpr char kTexture[] = "assets/wall.png";

    static constexpr float kBrickWidth = 2.0f;
    static constexpr float kBrickHeight = 2.0f;
    static constexpr float kBrickDepth = 0.25f;

    ~WidthWall() override = default;

    void AddDoor() final;
    void AddWindow() final;

protected:
    WidthWall(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);

private:
    static constexpr float kDoorDeltaX = 6.0f;
    static constexpr float kDoorDeltaY = 2.0f;

    static constexpr float kWindowDeltaX = 3.0f;
    static constexpr float kWindowDeltaY = 3.0f;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
