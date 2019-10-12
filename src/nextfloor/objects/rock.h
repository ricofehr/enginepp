/**
 *  @file rock.h
 *  @brief Rock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROCK_H_
#define NEXTFLOOR_OBJECTS_ROCK_H_

#include "nextfloor/objects/model_mesh.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/polygon_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Rock
 *  @brief Rock 3d model
 */
class Rock : public ModelMesh {

public:
    static constexpr float kBIG_SCALE = 1.0f;
    static constexpr float kMEDIUM_SCALE = 0.75f;
    static constexpr float kSMALL_SCALE = 0.5f;
    static constexpr float kSMALLER_SCALE = 0.5f;

    static constexpr char kTEXTURE[] = "assets/rock.jpg";

    Rock(std::unique_ptr<Border> border, std::vector<std::unique_ptr<Polygon>> rock, const glm::vec3& movement);
    ~Rock() final = default;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROCK_H_
