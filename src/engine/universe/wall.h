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

class Wall : public Model3D {

public:
    /* Face number */
    static const int kWALL_FRONT = 0;
    static const int kWALL_RIGHT = 1;
    static const int kWALL_BACK = 2;
    static const int kWALL_LEFT = 3;
    static const int kWALL_BOTTOM = 4;
    static const int kWALL_TOP = 5;

    Wall();
    Wall(int face, float scale, glm::vec4 location);

    /* Default move and copy constructor / operator */
    Wall(Wall&&) = default;
    Wall& operator=(Wall&&) = default;

    Wall(const Wall&) = default;
    Wall& operator=(const Wall&) = default;

    /* Default destructor */
    ~Wall() override = default;

private:
    int face_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_UNIVERSE_WALL_H_
