/*
 *  Wall class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Wall 3d model, inherits Model3D abstract class
 */

#ifndef ENGINE_UNIVERSE_WALL_H_
#define ENGINE_UNIVERSE_WALL_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {

namespace universe {

class Wall : public Model3D {

public:

    /*
     *  Side texture constants
     */
    static constexpr int kTEXTURE_TOP = 0;
    static constexpr int kTEXTURE_WALL = 1;
    static constexpr int kTEXTURE_FLOOR = 2;

    /* 
     *  Wall Side constants
     */
    static constexpr int kWALL_FRONT = 0;
    static constexpr int kWALL_RIGHT = 1;
    static constexpr int kWALL_BACK = 2;
    static constexpr int kWALL_LEFT = 3;
    static constexpr int kWALL_BOTTOM = 4;
    static constexpr int kWALL_TOP = 5;

    /*
     *  Constructors
     */
    Wall();
    Wall(glm::vec3 scale, glm::vec4 location, int face);

    /*
     *  Default move constructor and assignment
     */
    Wall(Wall&&) = default;
    Wall& operator=(Wall&&) = default;

    /*
     *  Default copy constructor and assignment
     */
    Wall(const Wall&) = default;
    Wall& operator=(const Wall&) = default;

    /*
     *  Default destructor
     */
    ~Wall() override = default;

    /*
     *  Fill vertex and texture into GL Buffers
     *  Needs a sole execution for all Walls object.
     */
    static void CreateBuffers();
};

} // namespace graphics

} // namespace engine

#endif // ENGINE_UNIVERSE_WALL_H_
