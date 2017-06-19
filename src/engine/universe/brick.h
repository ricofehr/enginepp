/**
 *  @file brick.h
 *  @brief Brick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef ENGINE_UNIVERSE_BRICK_H_
#define ENGINE_UNIVERSE_BRICK_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

/**
 *  @namespace engine
 *  @brief Common parent namespace
 */
namespace engine {

/**
 *  @namespace engine::universe
 *  @brief World elements
 */
namespace universe {

/**
 *  @class Brick
 *  @brief Brick 3d model, inherits Model3D abstract class
 */
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
