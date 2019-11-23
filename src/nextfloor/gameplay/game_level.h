/**
 *  @file game_level.h
 *  @brief Game Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_GAMELEVEL_H_
#define NEXTFLOOR_GAMEPLAY_GAMELEVEL_H_

#include "nextfloor/gameplay/level.h"

#include <memory>

#include "nextfloor/gameplay/renderer_factory.h"

#include "nextfloor/physic/collision_engine.h"
#include "nextfloor/character/character.h"
#include "nextfloor/playground/ground.h"
#include "nextfloor/thing/thing.h"
#include "nextfloor/character/camera.h"


namespace nextfloor {

namespace gameplay {

/**
 *  @class GameLevel
 *  @brief Define a determinitic demo level
 */
class GameLevel : public Level {

public:
    GameLevel(std::unique_ptr<nextfloor::playground::Ground> universe,
              std::unique_ptr<nextfloor::character::Character> player,
              std::unique_ptr<nextfloor::physic::CollisionEngine> collision_engine,
              RendererFactory* renderer_factory);
    ~GameLevel() final = default;

    GameLevel(GameLevel&&) = default;
    GameLevel& operator=(GameLevel&&) = default;
    GameLevel(const GameLevel&) = delete;
    GameLevel& operator=(const GameLevel&) = delete;

    void UpdateCameraOrientation(HIDPointer angles) final;
    void ExecutePlayerAction(Action* command, double elapsed_time) final;
    void Move() final;
    void Draw(float window_size_ratio) final;

private:
    void SetActiveCamera(nextfloor::character::Camera* active_camera);

    void DetectCollision(std::vector<nextfloor::mesh::Mesh*> moving_objects);
    void PivotCollisonOnObject(nextfloor::mesh::Mesh* pivot);
    void MoveObjects(std::vector<nextfloor::mesh::Mesh*> moving_objects);

    void PrepareDraw(float window_size_ratio);
    void Renderer(const nextfloor::mesh::Mesh& mesh);

    std::unique_ptr<nextfloor::playground::Ground> universe_{nullptr};
    nextfloor::character::Character* player_{nullptr};
    std::list<nextfloor::character::Camera*> game_cameras_;
    std::unique_ptr<nextfloor::physic::CollisionEngine> collision_engine_{nullptr};
    RendererFactory* renderer_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_GAMELEVEL_H_
