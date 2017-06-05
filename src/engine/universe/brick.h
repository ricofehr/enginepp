/*
* Brick class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_BRICK_H_
#define ENGINE_UNIVERSE_BRICK_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {
namespace universe {

/* Brick 3d model, inherit Model3D class */
class Brick : public Model3D {

public:

    Brick();
    Brick(float scale, glm::vec4 location);
    Brick(float scale, glm::vec4 location, glm::vec4 move);

    /* Default move and copy constructor / operator */
    Brick(Brick&&) = default;
    Brick& operator=(Brick&&) = default;

    Brick(const Brick&) = default;
    Brick& operator=(const Brick&) = default;

    /* Default destructor */
    ~Brick() override = default;

    static void CreateBuffers() noexcept;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_BRICK_H_
