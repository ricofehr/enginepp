/**
 *  @file factory.h
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_FACTORY_H_
#define NEXTFLOOR_FACTORY_FACTORY_H_

#include "nextfloor/factory/engine_factory.h"

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/factory/engine_renderer_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class Factory
 *  @brief Factory Pattern for Universe Objects
 */
class Factory : public EngineFactory {

public:

    Factory();

    Factory(Factory&&) = default;

    Factory& operator=(Factory&&) = default;

    Factory(const Factory&) = default;

    Factory& operator=(const Factory&) = default;

    virtual ~Factory() override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeUniverse() const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRoom(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeFrontWall(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRightWall(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeBackWall(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeLeftWall(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeFloor(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRoof(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRock(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeLittleRock(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeCamera(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EnginePolygon> MakeCube(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineBorder> MakeBorder(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeUniverseGrid(nextfloor::objects::EngineObject* universe) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeRoomGrid(nextfloor::objects::EngineObject* room) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeGrid(nextfloor::objects::EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeRoomGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* room_grid) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeUniverseGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* universe_grid) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* grid) const noexcept override;

    virtual nextfloor::objects::EngineRenderer* MakeCubeRenderer(std::string texture) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::EngineCollision> MakeCollisionEngine() const noexcept override;

private:

    //std::unique_ptr<nextfloor::objects::EngineObject> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept;

    std::unique_ptr<EngineRendererFactory> renderer_factory_;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_ENGINEFACTORY_H_
