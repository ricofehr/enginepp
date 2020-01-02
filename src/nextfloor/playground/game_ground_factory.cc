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
                                     nextfloor::mesh::BorderFactory* border_factory)
{
    thing_factory_ = thing_factory;
    grid_factory_ = grid_factory;
    border_factory_ = border_factory;
}

std::unique_ptr<Ground> GameGroundFactory::MakeUniverse(std::vector<std::unique_ptr<Ground>> rooms) const
{
    auto grid = grid_factory_->MakeUniverseGrid(glm::vec3(0.0f));
    auto border = border_factory_->MakeBorder(glm::vec3(0.0f), grid->scale());
    return std::make_unique<Universe>(std::move(grid), std::move(border), std::move(rooms));
}

std::unique_ptr<Ground> GameGroundFactory::MakeRoom(const glm::vec3& location,
                                                    std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects) const
{
    auto grid = grid_factory_->MakeRoomGrid(location);
    auto border = border_factory_->MakeBorder(location, grid->scale());

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
    auto brick_dimension = glm::vec3(FrontWall::kBrickWidth, FrontWall::kBrickHeight, FrontWall::kBrickDepth);
    return std::make_unique<FrontWall>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, FrontWall::kTexture));
}

std::unique_ptr<Wall> GameGroundFactory::MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto brick_dimension = glm::vec3(RightWall::kBrickWidth, RightWall::kBrickHeight, RightWall::kBrickDepth);
    return std::make_unique<RightWall>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, RightWall::kTexture));
}

std::unique_ptr<Wall> GameGroundFactory::MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto brick_dimension = glm::vec3(BackWall::kBrickWidth, BackWall::kBrickHeight, BackWall::kBrickDepth);
    return std::make_unique<BackWall>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, BackWall::kTexture));
}

std::unique_ptr<Wall> GameGroundFactory::MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const
{
    auto brick_dimension = glm::vec3(LeftWall::kBrickWidth, LeftWall::kBrickHeight, LeftWall::kBrickDepth);
    return std::make_unique<LeftWall>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, LeftWall::kTexture));
}

std::unique_ptr<Wall> GameGroundFactory::MakeFloor(const glm::vec3& location, const glm::vec3& scale) const
{
    auto brick_dimension = glm::vec3(Floor::kBrickWidth, Floor::kBrickHeight, Floor::kBrickDepth);
    return std::make_unique<Floor>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, Floor::kTexture));
}

std::unique_ptr<Wall> GameGroundFactory::MakeRoof(const glm::vec3& location, const glm::vec3& scale) const
{
    auto brick_dimension = glm::vec3(Roof::kBrickWidth, Roof::kBrickHeight, Roof::kBrickDepth);
    return std::make_unique<Roof>(
      thing_factory_->GenerateWallBricks(location - scale, location + scale, brick_dimension, Roof::kTexture));
}

}  // namespace playground

}  // namespace nextfloor
