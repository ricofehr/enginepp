/**
 *  @file brick.h
 *  @brief Brick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_BRICK_H_
#define NEXTFLOOR_OBJECTS_BRICK_H_

#include <glm/glm.hpp>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Brick
 *  @brief Brick 3d model, inherits Model3D abstract class
 */
class Brick : public Model {

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

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_BRICK_H_
