/**
 *  @file depth_wall.h
 *  @brief DepthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_DEPTHWALL_H_
#define NEXTFLOOR_OBJECTS_DEPTHWALL_H_

#include "nextfloor/objects/wall.h"

#include <glm/glm.hpp>
#include <string>

#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

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
    DepthWall(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks);
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_DEPTHWALL_H_
