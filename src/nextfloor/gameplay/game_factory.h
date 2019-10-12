/**
 *  @file gameplay_factory.h
 *  @brief Abstract Factory Class for gameplay part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_GAMEFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_GAMEFACTORY_H_

#include <memory>

#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/gameplay/level.h"
#include "nextfloor/gameplay/loop.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class GameFactory
 *  @brief Abstract Factory for common services
 */
class GameFactory {

public:
    virtual ~GameFactory() = default;

    virtual std::unique_ptr<Loop> MakeLoop() const = 0;
    virtual std::unique_ptr<Level> MakeLevel() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_GAMEFACTORY_H_
