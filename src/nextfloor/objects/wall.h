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

    static constexpr char kWALL_TEXTURE_FILE[] = "assets/wall.png";
    static constexpr char kSKY_TEXTURE_FILE[] = "assets/sky.png";
    static constexpr char kFLOOR_TEXTURE_FILE[] = "assets/floor.png";

    Wall(glm::vec3 scale, glm::vec4 location, std::unique_ptr<EngineRenderer> renderer);

    Wall(Wall&&) = default;

    Wall& operator=(Wall&&) = default;

    Wall(const Wall&) = delete;

    Wall& operator=(const Wall&) = delete;

    ~Wall() override = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
