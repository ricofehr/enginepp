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
                     RendererFactory* renderer_factory)
{
    player_ = player.get();
    universe_ = std::move(universe);
    universe_->AddIntoChild(std::move(player));
    game_cameras_ = universe_->all_cameras();
    SetActiveCamera(player_->camera());
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

void GameLevel::UpdateCameraOrientation(HIDPointer angles, float input_fov)
{
    auto active_camera = game_cameras_.front();
    active_camera->ComputeFOV(input_fov);
    active_camera->increment_angles(angles.horizontal_delta_angle, angles.vertical_delta_angle);
    active_camera->ComputeOrientation();
}

void GameLevel::ExecutePlayerAction(Action* command, double elapsed_time)
{
    command->execute(player_, elapsed_time);
}


void GameLevel::Move()
{
    universe_->DetectCollision();
    universe_->Move();
}

void GameLevel::Draw()
{
    auto active_camera = game_cameras_.front();
    universe_->PrepareDraw(*active_camera);
    auto polygons = universe_->GetPolygonsReadyToDraw(*active_camera);
    for (const auto& polygon : polygons) {
        auto renderer = renderer_factory_->MakeCubeRenderer(polygon->texture());
        renderer->Draw(polygon->mvp());
    }
}

}  // namespace gameplay

}  // namespace nextfloor
