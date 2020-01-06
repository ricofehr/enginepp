/**
 *  @file rock.h
 *  @brief Rock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_THING_ROCK_H_
#define NEXTFLOOR_THING_ROCK_H_

#include "nextfloor/thing/thing.h"

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/mesh/border.h"
#include "nextfloor/mesh/polygon.h"

namespace nextfloor {

namespace thing {

/**
 *  @class Rock
 *  @brief Rock 3d model
 */
class Rock : public Thing {

public:
    static constexpr char kTexture[] = "assets/rock.jpg";

    Rock(std::unique_ptr<nextfloor::mesh::Border> border,
         std::vector<std::unique_ptr<nextfloor::mesh::Polygon>> rock,
         const glm::vec3& movement);
    ~Rock() final = default;
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_THING_ROCK_H_
