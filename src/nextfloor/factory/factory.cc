/**
 *  @file factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/factory.h"

#include "nextfloor/factory/renderer_factory.h"

#include "nextfloor/objects/universe.h"
#include "nextfloor/objects/room.h"
#include "nextfloor/objects/front_wall.h"
#include "nextfloor/objects/right_wall.h"
#include "nextfloor/objects/back_wall.h"
#include "nextfloor/objects/left_wall.h"
#include "nextfloor/objects/floor.h"
#include "nextfloor/objects/roof.h"
#include "nextfloor/objects/wall_brick.h"
#include "nextfloor/objects/rock.h"
#include "nextfloor/objects/camera.h"

#include "nextfloor/grid/object_grid.h"
#include "nextfloor/grid/room_grid.h"
#include "nextfloor/grid/universe_grid.h"
#include "nextfloor/grid/room_grid_box.h"
#include "nextfloor/grid/universe_grid_box.h"
#include "nextfloor/grid/grid_box.h"

#include "nextfloor/polygons/cube.h"

#include "nextfloor/physics/border.h"
#include "nextfloor/physics/tbb_collision.h"
#include "nextfloor/physics/serial_collision.h"
#include "nextfloor/physics/cl_collision.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

Factory::Factory()
{
    assert(!sInstanciated);
    renderer_factory_ = std::make_unique<RendererFactory>();
    sInstanciated = true;
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeUniverse() const noexcept
{
    using nextfloor::objects::Universe;
    return std::make_unique<nextfloor::objects::Universe>();
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeRoom(glm::vec3 location) const noexcept
{
    using nextfloor::objects::Room;
    return std::make_unique<nextfloor::objects::Room>(location);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeFrontWall(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::FrontWall;
    return std::make_unique<nextfloor::objects::FrontWall>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeRightWall(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::RightWall;
    return std::make_unique<nextfloor::objects::RightWall>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeBackWall(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::BackWall;
    return std::make_unique<nextfloor::objects::BackWall>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeLeftWall(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::LeftWall;
    return std::make_unique<nextfloor::objects::LeftWall>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeFloor(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::Floor;
    return std::make_unique<nextfloor::objects::Floor>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeRoof(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::Roof;
    return std::make_unique<nextfloor::objects::Roof>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept
{
    using nextfloor::objects::WallBrick;
    return std::make_unique<nextfloor::objects::WallBrick>(location, scale, texture);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeRock(glm::vec3 location) const noexcept
{
    using nextfloor::objects::Rock;
    return std::make_unique<Rock>(location, 1.0f);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeLittleRock(glm::vec3 location) const noexcept
{
    using nextfloor::objects::Rock;
    return std::make_unique<Rock>(location, 0.5f);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeCamera(glm::vec3 location) const noexcept
{
    using nextfloor::objects::Camera;
    return std::make_unique<Camera>(location, 3.14f, 0.0f);
}

std::unique_ptr<nextfloor::objects::EnginePolygon> Factory::MakeCube(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::polygons::Cube;
    return std::make_unique<Cube>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineBorder> Factory::MakeBorder(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::physics::Border;
    return std::make_unique<Border>(location, scale);
}

std::unique_ptr<nextfloor::objects::EngineGrid> Factory::MakeUniverseGrid(nextfloor::objects::EngineObject* universe) const noexcept
{
    using nextfloor::grid::UniverseGrid;
    return std::make_unique<UniverseGrid>(universe);
}

std::unique_ptr<nextfloor::objects::EngineGrid> Factory::MakeRoomGrid(nextfloor::objects::EngineObject* room) const noexcept
{
    using nextfloor::grid::RoomGrid;
    return std::make_unique<RoomGrid>(room);
}

std::unique_ptr<nextfloor::objects::EngineGrid> Factory::MakeGrid(nextfloor::objects::EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension) const noexcept
{
    using nextfloor::grid::ObjectGrid;
    return std::make_unique<ObjectGrid>(owner, boxes_count, box_dimension);
}

std::unique_ptr<nextfloor::objects::EngineGridBox> Factory::MakeRoomGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* room_grid) const noexcept
{
    using nextfloor::grid::RoomGridBox;
    return std::make_unique<RoomGridBox>(grid_coords, room_grid);
}

std::unique_ptr<nextfloor::objects::EngineGridBox> Factory::MakeUniverseGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* universe_grid) const noexcept
{
    using nextfloor::grid::UniverseGridBox;
    return std::make_unique<UniverseGridBox>(grid_coords, universe_grid);
}

std::unique_ptr<nextfloor::objects::EngineGridBox> Factory::MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::EngineGrid* grid) const noexcept
{
    using nextfloor::grid::GridBox;
    return std::make_unique<GridBox>(grid_coords, grid);
}

nextfloor::objects::EngineRenderer* Factory::MakeCubeRenderer(std::string texture) const noexcept
{
    return renderer_factory_->MakeCubeRenderer(texture);
}

std::unique_ptr<nextfloor::objects::EngineCollision> Factory::MakeCollisionEngine() const noexcept
{
    using nextfloor::objects::EngineCollision;
    using nextfloor::core::CommonServices;
    using nextfloor::physics::Collision;
    using nextfloor::physics::TbbCollision;
    using nextfloor::physics::SerialCollision;
    using nextfloor::physics::ClCollision;

    std::unique_ptr<EngineCollision> engine_collision{nullptr};

    /* Get parallell type from config */
    int type_parallell = CommonServices::getConfig()->getParallellAlgoType();

    switch (type_parallell) {
        case Collision::kPARALLELL_TBB:
            engine_collision = std::make_unique<TbbCollision>();
            break;
        case Collision::kPARALLELL_CL:
            engine_collision = std::make_unique<ClCollision>();
            break;
        default:
            engine_collision = std::make_unique<SerialCollision>();
            break;
    }

    assert(engine_collision != nullptr);

    return engine_collision;
}


Factory::~Factory()
{
    sInstanciated = false;
}

} // namespace factory

} // namespace nextfloor