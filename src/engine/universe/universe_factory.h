/*
 *  Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
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

    /*
     *  Generate global GL Buffers
     *  Need executed only once and before any drawing object
     */
    virtual void GenerateBuffers() const = 0;

    /*
     *  Generate an Universe and return it
     */
    virtual std::unique_ptr<Universe> GenerateUniverse() const = 0;

    /*
     *  Generate one Room into Universe uni
     */
    virtual Room* GenerateRoom(Universe* uni) const = 0;

    /*
     *  Generate Walls 3d model into the Room r
     */
    virtual void GenerateWalls(Room* room) const = 0;

    /*
     *  Generate a Brick 3d model into the Room r
     */
    virtual void GenerateBrick(Room* room) const = 0;

    /*
     *  Generate a Camera Object and return it
     */
    virtual std::unique_ptr<Camera> GenerateCamera(glm::vec3 location) const = 0;

protected:

    /* Abstract Class, define destructor */
    virtual ~UniverseFactory() = default;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_UNIVERSEFACTORY_H_
