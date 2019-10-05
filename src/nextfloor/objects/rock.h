/**
 *  @file rock.h
 *  @brief Rock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROCK_H_
#define NEXTFLOOR_OBJECTS_ROCK_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class Rock
 *  @brief Rock 3d model
 */
class Rock : public ModelMesh {

public:
    Rock(const glm::vec3& location, float scale);
    ~Rock() final = default;

private:
    static constexpr char kTEXTURE[] = "assets/rock.jpg";
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROCK_H_
