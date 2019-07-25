/**
 *  @file demo_level.h
 *  @brief Demo Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_
#define NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_

#include "nextfloor/gameplay/level.h"

#include <memory>

#include "nextfloor/objects/engine_object.h"


namespace nextfloor {

namespace gameplay {

/**
 *  @class DemoLevel
 *  @brief Define a determinitic demo level
 */
class DemoLevel : public Level {

public:

    DemoLevel() = default;

    DemoLevel(DemoLevel&&) = default;
    DemoLevel& operator=(DemoLevel&&) = default;
    DemoLevel(const DemoLevel&) = delete;
    DemoLevel& operator=(const DemoLevel&) = delete;

    virtual ~DemoLevel() = default;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> GenerateUniverse() noexcept override;

};

} // namespace gameplay

} // namespace nextfloor

#endif // NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_
