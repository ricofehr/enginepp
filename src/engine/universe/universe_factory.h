/**
 *  Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *  Implements Factory Pattern for Universe Objects
 */

#ifndef ENGINE_UNIVERSE_UNIVERSEFACTORY_H_
#define ENGINE_UNIVERSE_UNIVERSEFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "engine/universe/universe.h"
#include "engine/universe/room.h"
#include "engine/universe/brick.h"
#include "engine/universe/wall.h"
#include "engine/universe/camera.h"

namespace engine {

namespace universe {

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
     *  Generate global GL Buffers
     *  Need executed only once and before any drawing object
     *  A pure virtual member.
     */
    virtual void GenerateBuffers() const = 0;

    /**
     *  Generate an Universe and return it
     *  A pure virtual member.
     *  @return the unique_ptr to the Universe created
     */
    virtual std::unique_ptr<Universe> GenerateUniverse() const = 0;

    /**
     *  Generate one Room into an Universe
     *  A pure virtual member.
     *  @param uni the universe targetted
     *  @return a raw pointer to the room created
     */
    virtual Room* GenerateRoom(Universe* uni) const = 0;

    /**
     *  Generate Walls 3d model into the Room room
     *  A pure virtual member.
     *  @param room is the place where the wall is created
     */
    virtual void GenerateWalls(Room* room) const = 0;

    /**
     *  Generate a Brick 3d model into the Room room
     *  A pure virtual member.
     *  @param room is the place where the brick is created
     */
    virtual void GenerateBrick(Room* room) const = 0;

    /**
     *  Generate a Camera Object and return it
     *  A pure virtual member.
     *  @param location is the coords of the camera
     *  @return an unique_ptr to the camera created
     */
    virtual std::unique_ptr<Camera> GenerateCamera(glm::vec3 location) const = 0;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    UniverseFactory() = default;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_UNIVERSEFACTORY_H_
