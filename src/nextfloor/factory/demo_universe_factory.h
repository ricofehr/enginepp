/**
 *  @file demo_universe_factory.h
 *  @brief Demo World Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_DEMOUNIVERSEFACTORY_H_
#define NEXTFLOOR_FACTORY_DEMOUNIVERSEFACTORY_H_

#include "nextfloor/factory/universe_factory.h"

#include <memory>
#include <glm/glm.hpp>


namespace nextfloor {

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

    virtual std::unique_ptr<nextfloor::objects::Universe> GenerateUniverse() const override final;

    virtual nextfloor::objects::Room* GenerateRoom(nextfloor::objects::Universe* universe) const override final;

    virtual void GenerateWalls(nextfloor::objects::Room* room) const override final;

    virtual void GenerateBrick(nextfloor::objects::Room* room) const override final;

    virtual std::unique_ptr<nextfloor::objects::Camera> GenerateCamera(glm::vec3 location) const override final;

private:

    void GenerateWallXFixed(nextfloor::objects::Room* room, int side) const;
    void GenerateWallYFixed(nextfloor::objects::Room* room, int side) const;
    void GenerateWallZFixed(nextfloor::objects::Room* room, int side) const;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_DEMOUNIVERSEFACTORY_H_
