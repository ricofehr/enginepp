/**
 *  @file floor.h
 *  @brief Floor class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_FLOOR_H_
#define NEXTFLOOR_OBJECTS_FLOOR_H_

#include "nextfloor/objects/wall.h"

#include <memory>
#include <glm/glm.hpp>
#include <string>

namespace nextfloor {

namespace objects {

/**
 *  @class Floor
 *  @brief Define Floor in a Room
 */
class Floor : public Wall {

public:
    static constexpr char kTEXTURE[] = "assets/floor.png";

    static constexpr float kBRICK_WIDTH = 2.0f;
    static constexpr float kBRICK_HEIGHT = 0.25f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    Floor(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> wall_bricks);
    ~Floor() final = default;

    void AddDoor() final;
    void AddWindow() final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
