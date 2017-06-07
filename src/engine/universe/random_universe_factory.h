/*
 *  Random World Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Generate Objects For Universe in Random manner
 */

#ifndef ENGINE_UNIVERSE_RANDOMUNIVERSEFACTORY_H_
#define ENGINE_UNIVERSE_RANDOMUNIVERSEFACTORY_H_

#include "engine/universe/universe_factory.h"

namespace engine {

namespace universe {

class RandomUniverseFactory : public UniverseFactory {

public:

    virtual ~RandomUniverseFactory() override = default;

    virtual void GenerateBuffers() const override final;
    virtual std::unique_ptr<Universe> GenerateUniverse() const override final;
    virtual Room* GenerateRoom(Universe* uni) const override final;
    virtual void GenerateWalls(Room* room) const override final;
    virtual void GenerateBrick(Room* room) const override final;
    virtual std::unique_ptr<Camera> GenerateCamera(glm::vec3 location) const override final;

};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_RANDOMUNIVERSEFACTORY_H_
