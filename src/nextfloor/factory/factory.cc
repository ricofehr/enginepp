/**
 *  @file factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/factory.h"


#include "nextfloor/objects/universe.h"
#include "nextfloor/objects/room.h"
#include "nextfloor/objects/wallbrick.h"
#include "nextfloor/objects/rock.h"
#include "nextfloor/objects/camera.h"

#include "nextfloor/grid/room_grid.h"
#include "nextfloor/grid/universe_grid.h"
#include "nextfloor/grid/room_grid_box.h"
#include "nextfloor/grid/universe_grid_box.h"

#include "nextfloor/polygons/cube.h"

#include "nextfloor/physics/border.h"
#include "nextfloor/physics/tbb_collision.h"
#include "nextfloor/physics/serial_collision.h"
#include "nextfloor/physics/cl_collision.h"

#include "nextfloor/renderer/gl_cube_renderer.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

Factory::Factory()
{
    assert(!sInstanciated);
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

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeFloorBrick(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::WallBrick;
    return MakeWallBrick(location, scale, WallBrick::kFLOOR_TEXTURE);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeSideWallBrick(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::WallBrick;
    return MakeWallBrick(location, scale, WallBrick::kWALL_TEXTURE);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeRoofBrick(glm::vec3 location, glm::vec3 scale) const noexcept
{
    using nextfloor::objects::WallBrick;
    return MakeWallBrick(location, scale, WallBrick::kROOF_TEXTURE);
}

std::unique_ptr<nextfloor::objects::EngineObject> Factory::MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept
{
    using nextfloor::objects::WallBrick;
    return std::make_unique<WallBrick>(location, scale, texture);
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

std::unique_ptr<nextfloor::objects::EngineRenderer> Factory::MakeCubeRenderer(std::string texture) const noexcept
{
    using nextfloor::renderer::GlCubeRenderer;
    return std::make_unique<GlCubeRenderer>(texture);
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