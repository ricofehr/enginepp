/**
 *  @file demo_level.h
 *  @brief Demo Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_RANDOMLEVEL_H_
#define NEXTFLOOR_GAMEPLAY_RANDOMLEVEL_H_

#include "nextfloor/gameplay/level.h"

#include <memory>

#include "nextfloor/objects/mesh.h"


namespace nextfloor {

namespace gameplay {

/**
 *  @class DemoLevel
 *  @brief Define a determinitic demo level
 */
class RandomLevel : public Level {

public:
    std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverse() const final;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_RANDOMLEVEL_H_
