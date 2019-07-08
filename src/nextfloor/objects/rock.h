/**
 *  @file brick.h
 *  @brief Brick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROCK_H_
#define NEXTFLOOR_OBJECTS_ROCK_H_

#include "nextfloor/objects/model.h"

#include <memory>

#include "nextfloor/objects/engine_renderer.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Rock
 *  @brief Rock 3d model, inherits Model3D abstract class
 */
class Rock : public Model {

public:

    static constexpr char kTEXTURE[] = "assets/rock.jpg";

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the brick
     *  @param renderer is the renderer object
     */
    Rock(glm::vec3 location, float scale);

    Rock(Rock&&) = default;
    Rock& operator=(Rock&&) = default;

    Rock(const Rock&) = delete;
    Rock& operator=(const Rock&) = delete;

    ~Rock() override = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ROCK_H_
