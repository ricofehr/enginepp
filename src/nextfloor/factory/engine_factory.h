/**
 *  @file engine_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_ENGINEFACTORY_H_
#define NEXTFLOOR_FACTORY_ENGINEFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_object.h"
#include "nextfloor/objects/engine_border.h"
#include "nextfloor/objects/engine_renderer.h"
#include "nextfloor/objects/engine_collision.h"
#include "nextfloor/objects/engine_grid.h"
#include "nextfloor/objects/engine_grid_box.h"
#include "nextfloor/objects/engine_polygon.h"

namespace nextfloor {

namespace factory {

/**
 *  @class EngineFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class EngineFactory {

public:

    EngineFactory(EngineFactory&&) = default;

    EngineFactory& operator=(EngineFactory&&) = default;

    EngineFactory(const EngineFactory&) = default;

    EngineFactory& operator=(const EngineFactory&) = default;

    virtual ~EngineFactory() = default;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeUniverse() const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRoom(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeFrontWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRightWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeBackWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeLeftWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeFloor(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRoof(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeRock(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeLittleRock(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineObject> MakeCamera(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EnginePolygon> MakeCube(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineBorder> MakeBorder(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeUniverseGrid(nextfloor::objects::EngineObject* universe) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeRoomGrid(nextfloor::objects::EngineObject* room) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGrid> MakeGrid(nextfloor::objects::EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeRoomGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* room_grid) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeUniverseGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* universe_grid) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* grid) const noexcept = 0;

    virtual nextfloor::objects::EngineRenderer* MakeCubeRenderer(std::string texture) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::EngineCollision> MakeCollisionEngine() const noexcept = 0;

protected:

    EngineFactory() = default;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_ENGINEFACTORY_H_
