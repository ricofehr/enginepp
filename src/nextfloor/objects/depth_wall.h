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
 *  @brief DepthWall 3d model
 */
class DepthWall : public Wall {

public:

    DepthWall(glm::vec3 location, glm::vec3 scale);

    DepthWall(DepthWall&&) = default;
    DepthWall& operator=(DepthWall&&) = default;

    DepthWall(const DepthWall&) = delete;
    DepthWall& operator=(const DepthWall&) = delete;

    ~DepthWall() override = default;

    virtual void AddDoor() noexcept override;

    virtual void AddWindow() noexcept override;

private:

    static constexpr char kTEXTURE[] = "assets/wall.png";

    static constexpr float kBRICK_WIDTH = 0.25;
    static constexpr float kBRICK_HEIGHT = 2.0f;
    static constexpr float kBRICK_DEPTH = 2.0f;

    virtual std::string texture_file() const noexcept override { return kTEXTURE; }
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_DEPTHWALL_H_
