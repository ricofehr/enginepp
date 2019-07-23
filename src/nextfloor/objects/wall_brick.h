/**
 *  @file wall_brick.h
 *  @brief WallBrick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WALLBRICK_H_
#define NEXTFLOOR_OBJECTS_WALLBRICK_H_

#include <glm/glm.hpp>
#include <string>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class WallBrick
 *  @brief Wall 3d model, inherits Model3D abstract class
 */
class WallBrick : public Model {

public:

    WallBrick(glm::vec3 location, glm::vec3 scale, std::string texture);

    WallBrick(WallBrick&&) = default;
    WallBrick& operator=(WallBrick&&) = default;

    WallBrick(const WallBrick&) = delete;
    WallBrick& operator=(const WallBrick&) = delete;

    ~WallBrick() override = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_OBJECTS_WALLBRICK_H_