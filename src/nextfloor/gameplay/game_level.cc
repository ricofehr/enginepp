/**
 *  @file demo_level.cc
 *  @brief DemoLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/game_level.h"

#include <tbb/tbb.h>
#include <list>
#include <utility>
#include <iterator>
#include <vector>
#include <string>

#include "nextfloor/gameplay/renderer_engine.h"

#include "nextfloor/mesh/mesh.h"
#include "nextfloor/character/camera.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

GameLevel::GameLevel(std::unique_ptr<nextfloor::playground::Ground> universe,
                     std::unique_ptr<nextfloor::character::Character> player,
                     std::unique_ptr<nextfloor::physic::CollisionEngine> collision_engine,
                     RendererFactory* renderer_factory)
{
    player_ = player.get();
    universe_ = std::move(universe);
    universe_->AddIntoChild(std::move(player));
    game_cameras_.push_front(player_->camera());
    SetActiveCamera(player_->camera());
    collision_engine_ = std::move(collision_engine);
    renderer_factory_ = renderer_factory;
}

void GameLevel::SetActiveCamera(nextfloor::character::Camera* active_camera)
{
    std::list<nextfloor::character::Camera*>::iterator it;
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

void GameLevel::ExecutePlayerAction(Action* command)
{
    command->execute(player_);
}

void GameLevel::UpdateCharacterStates(double elapsed_time)
{
    player_->UpdateState(elapsed_time);
}

void GameLevel::Move()
{
    std::vector<nextfloor::mesh::Mesh*> moving_objects = universe_->GetMovingObjects();
    DetectCollision(moving_objects);
    MoveObjects(moving_objects);
}

void GameLevel::DetectCollision(std::vector<nextfloor::mesh::Mesh*> moving_objects)
{
    tbb::parallel_for(0, (int)moving_objects.size(), 1, [&](int i) { PivotCollisonOnObject(moving_objects[i]); });
}

void GameLevel::PivotCollisonOnObject(nextfloor::mesh::Mesh* pivot)
{
    std::vector<nextfloor::mesh::Mesh*> test_objects = pivot->FindCollisionNeighbors();

    tbb::parallel_for(0, (int)test_objects.size(), 1, [&](int i) {
        assert(pivot->id() != test_objects[i]->id());
        collision_engine_->DetectCollision(pivot, test_objects[i]);
    });
}

void GameLevel::MoveObjects(std::vector<nextfloor::mesh::Mesh*> moving_objects)
{
    tbb::parallel_for(0, (int)moving_objects.size(), 1, [&](int i) { moving_objects[i]->MoveLocation(); });
}


void GameLevel::Draw(float window_size_ratio)
{
    PrepareDraw(window_size_ratio);
    Renderer(*universe_.get());
    RendererCubeMap(window_size_ratio);
}

void GameLevel::PrepareDraw(float window_size_ratio)
{
    nextfloor::character::Camera* active_camera = game_cameras_.front();
    universe_->PrepareDraw(active_camera->GetViewProjectionMatrix(window_size_ratio));
}

void GameLevel::Renderer(const nextfloor::mesh::Mesh& mesh)
{
    for (const auto& child : mesh.childs()) {
        Renderer(*child);
    }

    auto active_camera = game_cameras_.front();
    if (active_camera->IsInFieldOfView(mesh)) {
        std::vector<std::pair<glm::mat4, std::string>> mvps = mesh.GetModelViewProjectionsAndTextureToDraw();
        for (const auto& [mvp, texture] : mvps) {
            RendererEngine* renderer_engine = renderer_factory_->MakeCubeRenderer(texture);
            renderer_engine->Draw(mvp);
        }
    }
}

void GameLevel::RendererCubeMap(float window_size_ratio)
{
    RendererEngine* cube_map_renderer = renderer_factory_->MakeCubeMapRenderer();
    nextfloor::character::Camera* active_camera = game_cameras_.front();
    cube_map_renderer->Draw(active_camera->GetFarAndStaticViewProjectionMatrix(window_size_ratio));
}

}  // namespace gameplay

}  // namespace nextfloor
