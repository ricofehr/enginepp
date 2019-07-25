/**
 *  @file rock.h
 *  @brief Rock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROCK_H_
#define NEXTFLOOR_OBJECTS_ROCK_H_

#include "nextfloor/objects/model_mesh.h"

#include <memory>

#include "nextfloor/objects/renderer_engine.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Rock
 *  @brief Rock 3d model
 */
class Rock : public ModelMesh {

public:

    Rock(glm::vec3 location, float scale);

    Rock(Rock&&) = default;
    Rock& operator=(Rock&&) = default;
    Rock(const Rock&) = delete;
    Rock& operator=(const Rock&) = delete;

    virtual ~Rock() override = default;

private:

    static constexpr char kTEXTURE[] = "assets/rock.jpg";
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ROCK_H_
