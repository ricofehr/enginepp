/**
 *  @file brick.h
 *  @brief Brick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_BRICK_H_
#define NEXTFLOOR_OBJECTS_BRICK_H_

#include "nextfloor/objects/model.h"

#include "nextfloor/objects/engine_renderer.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Brick
 *  @brief Brick 3d model, inherits Model3D abstract class
 */
class Brick : public Model {

public:

    static constexpr char kTEXTURE_FILE[] = "assets/brique.png";

    /**
     *  Constructor
     *  @param scale is the float scale factor of native coords
     *  @param location is the center point of the brick
     *  @param renderer is the renderer object
     */
    Brick(float scale, glm::vec4 location, std::unique_ptr<EngineRenderer> renderer);

    Brick(Brick&&) = default;
    Brick& operator=(Brick&&) = default;

    Brick(const Brick&) = delete;
    Brick& operator=(const Brick&) = delete;

    ~Brick() override = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_BRICK_H_
