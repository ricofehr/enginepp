/**
 *  @file level.h
 *  @brief Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_LEVEL_H_
#define NEXTFLOOR_GAMEPLAY_LEVEL_H_

#include <memory>

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class Level
 *  @brief Abstract class who defines level interface
 */
class Level {

public:

    virtual ~Level() = default;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> GenerateUniverse() noexcept = 0;

protected:

    Level() = default;

    Level(Level&&) = default;
    Level& operator=(Level&&) = default;
    Level(const Level&) = delete;
    Level& operator=(const Level&) = delete;

};

} // namespace gameplay

} // namespace nextfloor

#endif // NEXTFLOOR_GAMEPLAY_LEVEL_H_
