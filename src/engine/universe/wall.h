/*
* Wall class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_WALL_H_
#define ENGINE_UNIVERSE_WALL_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {
namespace universe {

/* Wall 3d model, inherit Model3D class */
class Wall : public Model3D {

public:
    /* Face texture */
    static constexpr int kTEXTURE_TOP = 0;
    static constexpr int kTEXTURE_WALL = 1;
    static constexpr int kTEXTURE_FLOOR = 2;

    /* Face number */
    static const int kWALL_FRONT = 0;
    static const int kWALL_RIGHT = 1;
    static const int kWALL_BACK = 2;
    static const int kWALL_LEFT = 3;
    static const int kWALL_BOTTOM = 4;
    static const int kWALL_TOP = 5;

    Wall();
    Wall(glm::vec3 scale, glm::vec4 location, int face);

    /* Default move and copy constructor / operator */
    Wall(Wall&&) = default;
    Wall& operator=(Wall&&) = default;

    Wall(const Wall&) = default;
    Wall& operator=(const Wall&) = default;

    static void CreateBuffers();

    /* Default destructor */
    ~Wall() override = default;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_WALL_H_
