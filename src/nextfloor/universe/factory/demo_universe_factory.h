/**
 *  @file demo_universe_factory.h
 *  @brief Demo World Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_DEMOUNIVERSEFACTORY_H_
#define NEXTFLOOR_UNIVERSE_DEMOUNIVERSEFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/universe/factory/universe_factory.h"

namespace nextfloor {

namespace universe {

namespace factory {

/**
 *  @class DemoUniverseFactory
 *  @brief Implements the abstract UniverseFactory, generate Objects For Universe in Static manner for a sandbox demo
 */
class DemoUniverseFactory : public UniverseFactory {

public:

    DemoUniverseFactory() = default;

    DemoUniverseFactory(DemoUniverseFactory&&) = default;

    DemoUniverseFactory& operator=(DemoUniverseFactory&&) = default;

    DemoUniverseFactory(const DemoUniverseFactory&) = default;

    DemoUniverseFactory& operator=(const DemoUniverseFactory&) = default;

    virtual ~DemoUniverseFactory() override = default;

    virtual std::unique_ptr<nextfloor::universe::Universe> GenerateUniverse() const override final;

    virtual nextfloor::universe::Room* GenerateRoom(nextfloor::universe::Universe* universe) const override final;

    virtual void GenerateWalls(nextfloor::universe::Room* room) const override final;

    virtual void GenerateBrick(nextfloor::universe::Room* room) const override final;

    virtual std::unique_ptr<nextfloor::universe::dynamic::Camera> GenerateCamera(glm::vec4 location) const override final;

    virtual void GenerateBuffers() const override final;

private:

    void GenerateWallXFixed(nextfloor::universe::Room* room, int side) const;
    void GenerateWallYFixed(nextfloor::universe::Room* room, int side) const;
    void GenerateWallZFixed(nextfloor::universe::Room* room, int side) const;
};

} // namespace factory

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_DEMOUNIVERSEFACTORY_H_
