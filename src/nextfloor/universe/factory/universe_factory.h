/**
 *  @file universe_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_
#define NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/universe/universe.h"
#include "nextfloor/universe/room.h"
#include "nextfloor/universe/dynamic/brick.h"
#include "nextfloor/universe/stationary/wall.h"
#include "nextfloor/universe/dynamic/camera.h"

namespace nextfloor {

namespace universe {

namespace factory {

/**
 *  @class UniverseFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class UniverseFactory {

public:

    static constexpr int kUNIVERSEFACTORY_DEMO = 1;
    static constexpr int kUNIVERSEFACTORY_RANDOM = 2;

    UniverseFactory(UniverseFactory&&) = default;

    UniverseFactory& operator=(UniverseFactory&&) = default;

    UniverseFactory(const UniverseFactory&) = default;

    UniverseFactory& operator=(const UniverseFactory&) = default;

    virtual ~UniverseFactory() = default;

    virtual std::unique_ptr<nextfloor::universe::Universe> GenerateUniverse() const = 0;

    virtual nextfloor::universe::Room* GenerateRoom(nextfloor::universe::Universe* uni) const = 0;

    virtual void GenerateWalls(nextfloor::universe::Room* room) const = 0;

    virtual void GenerateBrick(nextfloor::universe::Room* room) const = 0;

    virtual std::unique_ptr<nextfloor::universe::dynamic::Camera> GenerateCamera(glm::vec4 location) const = 0;

    virtual void GenerateBuffers() const = 0;

protected:

    UniverseFactory() = default;
};

} // namespace factory

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_
