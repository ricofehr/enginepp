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

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::universe
 *  @brief World elements
 */
namespace universe {


/**
 *  @namespace nextfloor::universe::factory
 *  @brief World factory
 */
namespace factory {

/**
 *  @class UniverseFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class UniverseFactory {

public:

    /**
     *  Default Move constructor
     */
    UniverseFactory(UniverseFactory&&) = default;

    /**
     *  Default Move assignment
     */
    UniverseFactory& operator=(UniverseFactory&&) = default;

    /**
     *  Default Copy constructor
     */
    UniverseFactory(const UniverseFactory&) = default;

    /**
     *  Default Copy assignment
     */
    UniverseFactory& operator=(const UniverseFactory&) = default;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~UniverseFactory() = default;

    /**
     *  Generate an Universe and return it
     *  A pure virtual member.
     *  @return the unique_ptr to the Universe created
     */
    virtual std::unique_ptr<nextfloor::universe::Universe> GenerateUniverse() const = 0;

    /**
     *  Generate one Room into an Universe
     *  A pure virtual member.
     *  @param uni the universe targetted
     *  @return a raw pointer to the room created
     */
    virtual nextfloor::universe::Room* GenerateRoom(nextfloor::universe::Universe* uni) const = 0;

    /**
     *  Generate Walls 3d model into the Room room
     *  A pure virtual member.
     *  @param room is the place where the wall is created
     */
    virtual void GenerateWalls(nextfloor::universe::Room* room) const = 0;

    /**
     *  Generate a Brick 3d model into the Room room
     *  A pure virtual member.
     *  @param room is the place where the brick is created
     */
    virtual void GenerateBrick(nextfloor::universe::Room* room) const = 0;

    /**
     *  Generate a Camera Object and return it
     *  A pure virtual member.
     *  @param location is the coords of the camera
     *  @return an unique_ptr to the camera created
     */
    virtual std::unique_ptr<nextfloor::universe::dynamic::Camera> GenerateCamera(glm::vec4 location) const = 0;

    /**
     *  Generate global GL Buffers
     *  Need executed only once and before any drawing object
     *  A pure virtual member.
     */
    virtual void GenerateBuffers() const = 0;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    UniverseFactory() = default;
};

} // namespace factory

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_UNIVERSEFACTORY_H_
