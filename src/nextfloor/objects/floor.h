/**
 *  @file floor.h
 *  @brief Floor class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_FLOOR_H_
#define NEXTFLOOR_OBJECTS_FLOOR_H_

#include "nextfloor/objects/wall.h"

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
    Floor(const glm::vec3& location, const glm::vec3& scale);
    ~Floor() final = default;

    void AddDoor() final;
    void AddWindow() final;
    void PrepareDraw() final;

private:
    static constexpr char kTEXTURE[] = "assets/floor.png";

    static constexpr float kBRICK_WIDTH = 2.0f;
    static constexpr float kBRICK_HEIGHT = 0.25f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    std::string texture_file() const final { return kTEXTURE; }
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
