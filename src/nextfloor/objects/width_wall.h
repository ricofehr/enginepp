/**
 *  @file width_wall.h
 *  @brief WidthWall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WIDTHWALL_H_
#define NEXTFLOOR_OBJECTS_WIDTHWALL_H_

#include "nextfloor/objects/wall.h"

#include <glm/glm.hpp>
#include <string>


namespace nextfloor {

namespace objects {

/**
 *  @class WidthWall
 *  @brief WidthWall is an Abstract class for front / back walls of a Room
 */
class WidthWall : public Wall {

public:
    virtual ~WidthWall() override = default;

    virtual void AddDoor() noexcept override;
    virtual void AddWindow() noexcept override;

protected:
    WidthWall(glm::vec3 location, glm::vec3 scale);

    WidthWall(WidthWall&&) = default;
    WidthWall& operator=(WidthWall&&) = default;
    WidthWall(const WidthWall&) = delete;
    WidthWall& operator=(const WidthWall&) = delete;

private:
    static constexpr char kTEXTURE[] = "assets/wall.png";

    static constexpr float kBRICK_WIDTH = 2.0f;
    static constexpr float kBRICK_HEIGHT = 2.0f;
    static constexpr float kBRICK_DEPTH = 0.25f;

    virtual std::string texture_file() const noexcept override { return kTEXTURE; }
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
