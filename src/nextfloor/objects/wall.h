/**
 *  @file wall.h
 *  @brief Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WALL_H_
#define NEXTFLOOR_OBJECTS_WALL_H_

#include <glm/glm.hpp>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Wall
 *  @brief Wall 3d model, inherits Model3D abstract class
 */
class Wall : public Model {

public:

    static constexpr char kWALL_TEXTURE[] = "assets/wall.png";
    static constexpr char kSKY_TEXTURE[] = "assets/sky.png";
    static constexpr char kFLOOR_TEXTURE[] = "assets/floor.png";

    Wall(glm::vec3 scale, glm::vec4 location);

    Wall(Wall&&) = default;
    Wall& operator=(Wall&&) = default;

    Wall(const Wall&) = delete;
    Wall& operator=(const Wall&) = delete;

    ~Wall() override = default;

    void InitRenderer(std::unique_ptr<EngineRenderer> renderer) noexcept;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
