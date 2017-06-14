/**
 *   Brick class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
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

    /**
     *  Constructor
     */
    Brick();

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the brick
     */
    Brick(float scale, glm::vec4 location);

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the brick
     *  @param move is the translation vector
     */
    Brick(float scale, glm::vec4 location, glm::vec4 move);

    /**
     *  Default Move constructor
     */
    Brick(Brick&&) = default;

    /**
     *  Default Move assignment
     */
    Brick& operator=(Brick&&) = default;

    /**
     *  Copy constructor Deleted (Model3D Inherit)
     */
    Brick(const Brick&) = delete;

    /**
     *  Copy assignment Deleted (Model3D Inherit)
     */
    Brick& operator=(const Brick&) = delete;

    /**
     *  Default destructor
     */
    ~Brick() override = default;

    /**
     *  Fill vertex and texture into GL Buffers
     *  Needs a sole execution for all Bricks object.
     */
    static void CreateBuffers() noexcept;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_BRICK_H_
