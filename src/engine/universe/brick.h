/*
 *   Brick class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *   Brick 3d model, inherits Model3D abstract class
 */

#ifndef ENGINE_UNIVERSE_BRICK_H_
#define ENGINE_UNIVERSE_BRICK_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {

namespace universe {

class Brick : public Model3D {

public:

    /*
     *  Constructors
     */
    Brick();
    Brick(float scale, glm::vec4 location);
    Brick(float scale, glm::vec4 location, glm::vec4 move);

    /*
     *  Default move constructor and assignment
     */
    Brick(Brick&&) = default;
    Brick& operator=(Brick&&) = default;

    /*
     *  Default copy constructor and assignment
     */
    Brick(const Brick&) = default;
    Brick& operator=(const Brick&) = default;

    /*
     *  Default destructor
     */
    ~Brick() override = default;

    /*
     *  Fill vertex and texture into GL Buffers
     *  Needs a sole execution for all Bricks object.
     */
    static void CreateBuffers() noexcept;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_BRICK_H_
