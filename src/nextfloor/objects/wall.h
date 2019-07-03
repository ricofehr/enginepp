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

    /*
     *  Side texture constants
     */
    static constexpr int kTEXTURE_TOP = 0;
    static constexpr int kTEXTURE_WALL = 1;
    static constexpr int kTEXTURE_FLOOR = 2;

    Wall(glm::vec3 scale, glm::vec4 location, int face);

    Wall(Wall&&) = default;

    Wall& operator=(Wall&&) = default;

    Wall(const Wall&) = delete;

    Wall& operator=(const Wall&) = delete;

    ~Wall() override = default;

    /**
     *  Fill vertex and texture into GL Buffers
     *  Needs a sole execution for all Walls object.
     */
    static void CreateBuffers();
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_WALL_H_
