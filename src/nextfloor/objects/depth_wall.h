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

namespace nextfloor {

namespace objects {

/**
 *  @class DepthWall
 *  @brief DepthWall is an Abstract class for left / right wall of a Room
 */
class DepthWall : public Wall {

public:
    ~DepthWall() override = default;

    void AddDoor() final;
    void AddWindow() final;

protected:
    DepthWall(const glm::vec3& location, const glm::vec3& scale);

private:
    static constexpr char kTEXTURE[] = "assets/wall.png";

    static constexpr float kBRICK_WIDTH = 0.25;
    static constexpr float kBRICK_HEIGHT = 2.0f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    std::string texture_file() const final { return kTEXTURE; }
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_DEPTHWALL_H_
