/**
 *  @file universe_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_
#define NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/universe.h"
#include "nextfloor/objects/room.h"
#include "nextfloor/objects/wall.h"
#include "nextfloor/objects/brick.h"
#include "nextfloor/objects/camera.h"

namespace nextfloor {

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

    virtual std::unique_ptr<nextfloor::objects::Universe> GenerateUniverse() const = 0;

    virtual nextfloor::objects::Room* GenerateRoom(nextfloor::objects::Universe* uni) const = 0;

    virtual void GenerateWalls(nextfloor::objects::Room* room) const = 0;

    virtual void GenerateBrick(nextfloor::objects::Room* room) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Camera> GenerateCamera(glm::vec3 location) const = 0;

protected:

    UniverseFactory() = default;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_UNIVERSEFACTORY_H_
