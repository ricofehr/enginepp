/**
 *  @file floor.h
 *  @brief Floor class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOF_H_
#define NEXTFLOOR_OBJECTS_ROOF_H_

#include "nextfloor/objects/wall.h"

#include <memory>
#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class Wall
 *  @brief Wall 3d model
 */
class Roof : public Wall {

public:
    static constexpr char kTEXTURE[] = "assets/sky.png";

    static constexpr float kBRICK_WIDTH = 2.0f;
    static constexpr float kBRICK_HEIGHT = 0.25f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    Roof(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks);
    ~Roof() final = default;

    void AddDoor() final;
    void AddWindow() final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
