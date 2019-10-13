/**
 *  @file demo_level.cc
 *  @brief DemoLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/game_level.h"

#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

GameLevel::GameLevel(std::unique_ptr<nextfloor::objects::Mesh> universe,
                     std::unique_ptr<nextfloor::objects::Mesh> player,
                     std::unique_ptr<nextfloor::objects::CollisionEngine> collision_engine,
                     RendererFactory* renderer_factory)
{
    player_ = player.get();
    universe_ = std::move(universe);
    universe_->AddIntoChild(std::move(player));
    game_cameras_ = universe_->all_cameras();
    SetActiveCamera(player_->camera());
    collision_engine_ = std::move(collision_engine);
    renderer_factory_ = renderer_factory;
}

void GameLevel::SetActiveCamera(nextfloor::objects::Camera* active_camera)
{
    std::list<nextfloor::objects::Camera*>::iterator it;
    for (it = game_cameras_.begin(); it != game_cameras_.end(); ++it) {
        if (*it == active_camera) {
            game_cameras_.remove(active_camera);
            game_cameras_.push_front(active_camera);
            break;
        }
    }
}

void GameLevel::UpdateCameraOrientation(HIDPointer angles)
{
    auto active_camera = game_cameras_.front();
    active_camera->increment_angles(angles.horizontal_delta_angle, angles.vertical_delta_angle);
    active_camera->ComputeOrientation();
}

void GameLevel::ExecutePlayerAction(Action* command, double elapsed_time)
{
    command->execute(player_, elapsed_time);
}


void GameLevel::Move()
{
    auto moving_objects = universe_->GetMovingObjects();
    DetectCollision(moving_objects);
    MoveObjects(moving_objects);
}

void GameLevel::DetectCollision(std::vector<nextfloor::objects::Mesh*> moving_objects)
{
    tbb::parallel_for(0, (int)moving_objects.size(), 1, [&](int i) { PivotCollisonOnObject(moving_objects[i]); });
}

void GameLevel::PivotCollisonOnObject(nextfloor::objects::Mesh* pivot)
{
    auto test_objects = pivot->FindCollisionNeighbors();

    tbb::parallel_for(0, (int)test_objects.size(), 1, [&](int i) {
        assert(pivot->id() != test_objects[i]->id());
        collision_engine_->DetectCollision(pivot, test_objects[i]);
    });
}

void GameLevel::MoveObjects(std::vector<nextfloor::objects::Mesh*> moving_objects)
{
    tbb::parallel_for(0, (int)moving_objects.size(), 1, [&](int i) {
        moving_objects[i]->MoveLocation();
        moving_objects[i]->UpdateGridPlacement();
    });
}


void GameLevel::Draw(float window_size_ratio)
{
    auto active_camera = game_cameras_.front();
    universe_->PrepareDraw(active_camera->GetViewProjectionMatrix(window_size_ratio));
    auto polygons = universe_->GetPolygonsReadyToDraw(*active_camera);
    for (const auto& polygon : polygons) {
        auto renderer = renderer_factory_->MakeCubeRenderer(polygon->texture());
        renderer->Draw(polygon->mvp());
    }
}

}  // namespace gameplay

}  // namespace nextfloor
