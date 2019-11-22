/**
 *  @file model_mesh_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/playground/game_ground_factory.h"

#include "nextfloor/playground/universe.h"
#include "nextfloor/playground/room.h"
#include "nextfloor/playground/front_wall.h"
#include "nextfloor/playground/right_wall.h"
#include "nextfloor/playground/back_wall.h"
#include "nextfloor/playground/left_wall.h"
#include "nextfloor/playground/floor.h"
#include "nextfloor/playground/roof.h"

namespace nextfloor {

namespace playground {

GameGroundFactory::GameGroundFactory(nextfloor::thing::ThingFactory* thing_factory,
                                     GridFactory* grid_factory,
                                     nextfloor::mesh::PhysicFactory* physic_factory)
{
    thing_factory_ = thing_factory;
    grid_factory_ = grid_factory;
    physic_factory_ = physic_factory;
}

std::unique_ptr<Ground> GameGroundFactory::MakeUniverse(std::vector<std::unique_ptr<Ground>> rooms) const
{
    auto grid = grid_factory_->MakeUniverseGrid(glm::vec3(0.0f));
    auto border = physic_factory_->MakeBorder(glm::vec3(0.0f), grid->scale());
    return std::make_unique<Universe>(std::move(grid), std::move(border), std::move(rooms));
}

std::unique_ptr<Ground> GameGroundFactory::MakeRoom(const glm::vec3& location,
                                                    std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects) const
{
    auto grid = grid_factory_->MakeRoomGrid(location);
    auto border = physic_factory_->MakeBorder(location, grid->scale());

    std::vector<std::unique_ptr<Wall>> walls;
    walls.push_back(MakeFrontWall(grid->CalculateFrontSideLocation(), grid->CalculateFrontSideBorderScale()));
    walls.push_back(MakeRightWall(grid->CalculateRightSideLocation(), grid->CalculateRightSideBorderScale()));
    walls.push_back(MakeBackWall(grid->CalculateBackSideLocation(), grid->CalculateBackSideBorderScale()));
    walls.push_back(MakeLeftWall(grid->CalculateLeftSideLocation(), grid->CalculateLeftSideBorderScale()));
    walls.push_back(MakeFloor(grid->CalculateBottomSideLocation(), grid->CalculateBottomSideBorderScale()));
    walls.push_back(MakeRoof(grid->CalculateTopSideLocation(), grid->CalculateTopSideBorderScale()));

    return std::make_unique<Room>(std::move(grid), std::move(border), std::move(walls), std::move(objects));
}

std::unique_ptr<Wall> GameGroundFactory::MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(FrontWall::kBRICK_WIDTH, FrontWall::kBRICK_HEIGHT, FrontWall::kBRICK_DEPTH);
    return std::make_unique<FrontWall>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, FrontWall::kTEXTURE));
}

std::unique_ptr<Wall> GameGroundFactory::MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(RightWall::kBRICK_WIDTH, RightWall::kBRICK_HEIGHT, RightWall::kBRICK_DEPTH);
    return std::make_unique<RightWall>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, RightWall::kTEXTURE));
}

std::unique_ptr<Wall> GameGroundFactory::MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(BackWall::kBRICK_WIDTH, BackWall::kBRICK_HEIGHT, BackWall::kBRICK_DEPTH);
    return std::make_unique<BackWall>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, BackWall::kTEXTURE));
}

std::unique_ptr<Wall> GameGroundFactory::MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(LeftWall::kBRICK_WIDTH, LeftWall::kBRICK_HEIGHT, LeftWall::kBRICK_DEPTH);
    return std::make_unique<LeftWall>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, LeftWall::kTEXTURE));
}

std::unique_ptr<Wall> GameGroundFactory::MakeFloor(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(Floor::kBRICK_WIDTH, Floor::kBRICK_HEIGHT, Floor::kBRICK_DEPTH);
    return std::make_unique<Floor>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, Floor::kTEXTURE));
}

std::unique_ptr<Wall> GameGroundFactory::MakeRoof(const glm::vec3& location, const glm::vec3& scale) const
{
    auto border = physic_factory_->MakeBorder(location, scale);
    auto brick_dimension = glm::vec3(Roof::kBRICK_WIDTH, Roof::kBRICK_HEIGHT, Roof::kBRICK_DEPTH);
    return std::make_unique<Roof>(
      std::move(border),
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, Roof::kTEXTURE));
}

}  // namespace playground

}  // namespace nextfloor
