/**
 *  @file game_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/objects/model_mesh_factory.h"

#include "nextfloor/objects/universe.h"
#include "nextfloor/objects/room.h"
#include "nextfloor/objects/front_wall.h"
#include "nextfloor/objects/right_wall.h"
#include "nextfloor/objects/back_wall.h"
#include "nextfloor/objects/left_wall.h"
#include "nextfloor/objects/floor.h"
#include "nextfloor/objects/roof.h"
#include "nextfloor/objects/wall_brick.h"
#include "nextfloor/gameplay/player.h"
#include "nextfloor/objects/rock.h"
#include "nextfloor/gameplay/head_camera.h"

#include "nextfloor/grid/mesh_grid.h"
#include "nextfloor/grid/room_grid.h"
#include "nextfloor/grid/universe_grid.h"
#include "nextfloor/grid/room_grid_box.h"
#include "nextfloor/grid/universe_grid_box.h"
#include "nextfloor/grid/wired_grid_box.h"

#include "nextfloor/polygons/cube.h"

#include "nextfloor/physics/cube_border.h"
#include "nextfloor/physics/tbb_nearer_collision_engine.h"
#include "nextfloor/physics/serial_nearer_collision_engine.h"
#include "nextfloor/physics/cl_nearer_collision_engine.h"

#include "nextfloor/gameplay/demo_level.h"

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace objects {

std::unique_ptr<Mesh> ModelMeshFactory::MakeUniverse() const
{
    return std::make_unique<Universe>(*this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRoom(const glm::vec3& location) const
{
    return std::make_unique<Room>(location, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<FrontWall>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<RightWall>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<BackWall>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<LeftWall>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeFloor(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<Floor>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRoof(const glm::vec3& location, const glm::vec3& scale) const
{
    return std::make_unique<Roof>(location, scale, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeWallBrick(const glm::vec3& location,
                                                      const glm::vec3& scale,
                                                      const std::string& texture) const
{
    return std::make_unique<WallBrick>(location, scale, texture, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakePlayer(const glm::vec3& location) const
{
    using nextfloor::gameplay::Player;
    return std::make_unique<Player>(location, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRock(const glm::vec3& location) const
{
    return std::make_unique<Rock>(location, 1.0f, *this);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeLittleRock(const glm::vec3& location) const
{
    return std::make_unique<Rock>(location, 0.5f, *this);
}

std::unique_ptr<Camera> ModelMeshFactory::MakeCamera(Mesh* owner) const
{
    using nextfloor::gameplay::HeadCamera;
    return std::make_unique<HeadCamera>(owner, 3.14f, 0.0f);
}

std::unique_ptr<Polygon> ModelMeshFactory::MakeCube(const glm::vec3& location, const glm::vec3& scale) const
{
    using nextfloor::polygons::Cube;
    return std::make_unique<Cube>(location, scale);
}

std::unique_ptr<Polygon> ModelMeshFactory::MakeCube(const glm::vec3& location,
                                                    const glm::vec3& scale,
                                                    const std::string& texture) const
{
    using nextfloor::polygons::Cube;
    return std::make_unique<Cube>(location, scale, texture);
}

std::unique_ptr<Border> ModelMeshFactory::MakeBorder(const glm::vec3& location, const glm::vec3& scale) const
{
    using nextfloor::physics::CubeBorder;
    return std::make_unique<CubeBorder>(location, scale, *this);
}

std::unique_ptr<Grid> ModelMeshFactory::MakeUniverseGrid(Mesh* universe) const
{
    using nextfloor::grid::UniverseGrid;
    return std::make_unique<UniverseGrid>(universe, *this);
}

std::unique_ptr<Grid> ModelMeshFactory::MakeRoomGrid(Mesh* room) const
{
    using nextfloor::grid::RoomGrid;
    return std::make_unique<RoomGrid>(room, *this);
}

std::unique_ptr<Grid> ModelMeshFactory::MakeGrid(Mesh* owner,
                                                 const glm::ivec3& boxes_count,
                                                 const glm::vec3& box_dimension) const
{
    using nextfloor::grid::MeshGrid;
    return std::make_unique<MeshGrid>(owner, boxes_count, box_dimension, *this);
}

std::unique_ptr<GridBox> ModelMeshFactory::MakeRoomGridBox(const glm::vec3& coords, Grid* room_grid) const
{
    using nextfloor::grid::RoomGridBox;
    return std::make_unique<RoomGridBox>(coords, room_grid);
}

std::unique_ptr<GridBox> ModelMeshFactory::MakeUniverseGridBox(const glm::vec3& coords, Grid* universe_grid) const
{
    using nextfloor::grid::UniverseGridBox;
    return std::make_unique<UniverseGridBox>(coords, universe_grid);
}

std::unique_ptr<GridBox> ModelMeshFactory::MakeGridBox(const glm::vec3& grid_coords, Grid* grid) const
{
    using nextfloor::grid::WiredGridBox;
    return std::make_unique<WiredGridBox>(grid_coords, grid);
}

CollisionEngine* ModelMeshFactory::MakeCollisionEngine() const
{
    using nextfloor::core::CommonServices;
    using nextfloor::physics::ClNearerCollisionEngine;
    using nextfloor::physics::NearerCollisionEngine;
    using nextfloor::physics::SerialNearerCollisionEngine;
    using nextfloor::physics::TbbNearerCollisionEngine;

    CollisionEngine* engine_collision{nullptr};

    /* Get parallell type from config */
    int type_parallell = CommonServices::getConfig()->getParallellAlgoType();

    switch (type_parallell) {  // clang-format off
        case NearerCollisionEngine::kPARALLELL_TBB:
            engine_collision = new TbbNearerCollisionEngine();
            break;
        case NearerCollisionEngine::kPARALLELL_CL:
            engine_collision = new ClNearerCollisionEngine();
            break;
        default:
            engine_collision = new SerialNearerCollisionEngine();
            break;
    }  // clang-format on

    assert(engine_collision != nullptr);

    return engine_collision;
}

std::unique_ptr<nextfloor::gameplay::Level> ModelMeshFactory::MakeLevel() const
{
    using nextfloor::gameplay::DemoLevel;
    return std::make_unique<DemoLevel>(*this);
}

}  // namespace objects

}  // namespace nextfloor