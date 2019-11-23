/**
 *  @file floor.h
 *  @brief Floor class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_FLOOR_H_
#define NEXTFLOOR_PLAYGROUND_FLOOR_H_

#include "nextfloor/playground/wall.h"

#include <memory>
#include <glm/glm.hpp>
#include <string>

namespace nextfloor {

namespace playground {

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

    Floor(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks);
    ~Floor() final = default;

    void AddDoor() final;
    void AddWindow() final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) final;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
