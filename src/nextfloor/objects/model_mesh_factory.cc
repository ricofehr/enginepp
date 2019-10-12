/**
 *  @file model_mesh_factory.c
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
#include "nextfloor/objects/rock.h"

namespace nextfloor {

namespace objects {

ModelMeshFactory::ModelMeshFactory(PolygonFactory* polygon_factory, GridFactory* grid_factory, PhysicFactory* physic_factory)
{
    polygon_factory_ = polygon_factory;
    grid_factory_ = grid_factory;
    physic_factory_ = physic_factory;
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeUniverse() const
{
    auto grid = grid_factory_->MakeUniverseGrid(glm::vec3(0.0f));
    auto border = physic_factory_->MakeBorder(glm::vec3(0.0f), grid->scale());
    return std::make_unique<Universe>(std::move(grid), std::move(border));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRoom(const glm::vec3& location) const
{
    auto grid = grid_factory_->MakeRoomGrid(location);
    auto border = physic_factory_->MakeBorder(location, grid->scale());

    std::vector<std::unique_ptr<Mesh>> childs;
    childs.push_back(MakeFrontWall(grid->CalculateFrontSideLocation(), grid->CalculateFrontSideBorderScale()));
    childs.push_back(MakeRightWall(grid->CalculateRightSideLocation(), grid->CalculateRightSideBorderScale()));
    childs.push_back(MakeBackWall(grid->CalculateBackSideLocation(), grid->CalculateBackSideBorderScale()));
    childs.push_back(MakeLeftWall(grid->CalculateLeftSideLocation(), grid->CalculateLeftSideBorderScale()));
    childs.push_back(MakeFloor(grid->CalculateBottomSideLocation(), grid->CalculateBottomSideBorderScale()));
    childs.push_back(MakeRoof(grid->CalculateTopSideLocation(), grid->CalculateTopSideBorderScale()));

    return std::make_unique<Room>(std::move(grid), std::move(border), std::move(childs));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(FrontWall::kBRICK_WIDTH, FrontWall::kBRICK_HEIGHT, FrontWall::kBRICK_DEPTH);
    return std::make_unique<FrontWall>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, FrontWall::kTEXTURE));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(RightWall::kBRICK_WIDTH, RightWall::kBRICK_HEIGHT, RightWall::kBRICK_DEPTH);
    return std::make_unique<RightWall>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, RightWall::kTEXTURE));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(BackWall::kBRICK_WIDTH, BackWall::kBRICK_HEIGHT, BackWall::kBRICK_DEPTH);
    return std::make_unique<BackWall>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, BackWall::kTEXTURE));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(LeftWall::kBRICK_WIDTH, LeftWall::kBRICK_HEIGHT, LeftWall::kBRICK_DEPTH);
    return std::make_unique<LeftWall>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, LeftWall::kTEXTURE));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeFloor(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(Floor::kBRICK_WIDTH, Floor::kBRICK_HEIGHT, Floor::kBRICK_DEPTH);
    return std::make_unique<Floor>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, Floor::kTEXTURE));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRoof(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(Roof::kBRICK_WIDTH, Roof::kBRICK_HEIGHT, Roof::kBRICK_DEPTH);
    return std::make_unique<Roof>(
      std::move(border), GenerateWallBricks(location - scale, location + scale, brick_dimension, Roof::kTEXTURE));
}

std::vector<std::unique_ptr<Mesh>> ModelMeshFactory::GenerateWallBricks(glm::vec3 firstpoint,
                                                                        glm::vec3 lastpoint,
                                                                        const glm::vec3& brick_dimension,
                                                                        const std::string& texture) const
{
    auto padding = brick_dimension / 2.0f;
    std::vector<std::unique_ptr<Mesh>> bricks;
    firstpoint += padding;
    lastpoint -= padding;

    for (float x = firstpoint.x; x <= lastpoint.x; x += brick_dimension.x) {
        for (float y = firstpoint.y; y <= lastpoint.y; y += brick_dimension.y) {
            for (float z = firstpoint.z; z <= lastpoint.z; z += brick_dimension.z) {
                auto location = glm::vec3(x, y, z);
                bricks.push_back(MakeWallBrick(location, brick_dimension / 2.0f, texture));
            }
        }
    }

    return bricks;
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeWallBrick(const glm::vec3& location,
                                                      const glm::vec3& scale,
                                                      const std::string& texture) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    std::vector<std::unique_ptr<Polygon>> bricks;
    bricks.push_back(polygon_factory_->MakeCube(location, scale, texture));
    return std::make_unique<WallBrick>(std::move(border), std::move(bricks));
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeRock(const glm::vec3& location, const glm::vec3& movement) const
{
    auto border = physic_factory_->MakeBorder(location, Rock::kBIG_SCALE);
    std::vector<std::unique_ptr<Polygon>> rocks;
    rocks.push_back(polygon_factory_->MakeCube(location, Rock::kBIG_SCALE, Rock::kTEXTURE));
    return std::make_unique<Rock>(std::move(border), std::move(rocks), movement);
}

std::unique_ptr<Mesh> ModelMeshFactory::MakeLittleRock(const glm::vec3& location, const glm::vec3& movement) const
{
    auto border = physic_factory_->MakeBorder(location, Rock::kSMALL_SCALE);
    std::vector<std::unique_ptr<Polygon>> rocks;
    rocks.push_back(polygon_factory_->MakeCube(location, Rock::kSMALL_SCALE, Rock::kTEXTURE));
    return std::make_unique<Rock>(std::move(border), std::move(rocks), movement);
}

}  // namespace objects

}  // namespace nextfloor