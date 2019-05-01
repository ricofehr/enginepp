/**
 *  @file random_universe_factory.h
 *  @brief Random World Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_RANDOMUNIVERSEFACTORY_H_
#define NEXTFLOOR_UNIVERSE_RANDOMUNIVERSEFACTORY_H_

#include "nextfloor/universe/factory/universe_factory.h"

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
 *  @class RandomUniverseFactory
 *  @brief Implements the abstract UniverseFactory, generate Objects For Universe in Random manner
 */
class RandomUniverseFactory : public UniverseFactory {

public:

    /**
     *  Default Constructor
     */
    RandomUniverseFactory() = default;

    /**
     *  Default Move constructor
     */
    RandomUniverseFactory(RandomUniverseFactory&&) = default;

    /**
     *  Default Move assignment
     */
    RandomUniverseFactory& operator=(RandomUniverseFactory&&) = default;

    /**
     *  Default Copy constructor
     */
    RandomUniverseFactory(const RandomUniverseFactory&) = default;

    /**
     *  Default Copy assignment
     */
    RandomUniverseFactory& operator=(const RandomUniverseFactory&) = default;

    /**
     *  Default destructor
     */
    virtual ~RandomUniverseFactory() override = default;

    /**
     *  Generate an Universe and return it
     *  @return the unique_ptr to the Universe created
     */
    virtual std::unique_ptr<nextfloor::universe::Universe> GenerateUniverse() const override final;

    /**
     *  Generate one Room into an Universe
     *  @param uni the universe targetted
     *  @return a raw pointer to the room created
     */
    virtual nextfloor::universe::Room* GenerateRoom(nextfloor::universe::Universe* uni) const override final;

    /**
     *  Generate Walls 3d model into the Room room
     *  @param room is the place where the wall is created
     */
    virtual void GenerateWalls(nextfloor::universe::Room* room) const override final;

    /**
     *  Generate a Brick 3d model into the Room room
     *  @param room is the place where the brick is created
     */
    virtual void GenerateBrick(nextfloor::universe::Room* room) const override final;

    /**
     *  Generate a Camera Object and return it
     *  @param location is the coords of the camera
     *  @return an unique_ptr to the camera created
     */
    virtual std::unique_ptr<nextfloor::universe::dynamic::Camera> GenerateCamera(glm::vec4 location) const override final;

    /**
     *  Generate global GL Buffers
     *  Need executed only once and before any drawing object
     */
    virtual void GenerateBuffers() const override final;

};

} // namespace factory

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_RANDOMUNIVERSEFACTORY_H_
