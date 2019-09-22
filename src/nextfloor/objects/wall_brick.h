/**
 *  @file wall_brick.h
 *  @brief WallBrick class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_WALLBRICK_H_
#define NEXTFLOOR_OBJECTS_WALLBRICK_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>


namespace nextfloor {

namespace objects {

/**
 *  @class WallBrick
 *  @brief WallBrick model, inherits Model abstract class
 */
class WallBrick : public ModelMesh {

public:
    WallBrick(const glm::vec3& location, const glm::vec3& scale, const std::string& texture);
    ~WallBrick() final = default;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_UNIVERSE_OBJECTS_WALLBRICK_H_