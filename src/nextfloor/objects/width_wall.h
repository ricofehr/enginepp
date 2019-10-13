/**
 *  @file width_wall.h
 *  @brief WidthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WIDTHWALL_H_
#define NEXTFLOOR_OBJECTS_WIDTHWALL_H_

#include "nextfloor/objects/wall.h"

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/border.h"

namespace nextfloor {

namespace objects {

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
    WidthWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks);
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
