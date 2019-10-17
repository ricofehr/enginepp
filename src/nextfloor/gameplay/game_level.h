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

#include "nextfloor/gameplay/collision_engine.h"
#include "nextfloor/gameplay/character.h"
#include "nextfloor/objects/mesh.h"
#include "nextfloor/gameplay/camera.h"


namespace nextfloor {

namespace gameplay {

/**
 *  @class GameLevel
 *  @brief Define a determinitic demo level
 */
class GameLevel : public Level {

public:
    GameLevel(std::unique_ptr<nextfloor::objects::Mesh> universe,
              std::unique_ptr<Character> player,
              std::unique_ptr<CollisionEngine> collision_engine,
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
    void SetActiveCamera(Camera* active_camera);

    void DetectCollision(std::vector<nextfloor::objects::Mesh*> moving_objects);
    void PivotCollisonOnObject(nextfloor::objects::Mesh* pivot);
    void MoveObjects(std::vector<nextfloor::objects::Mesh*> moving_objects);

    void PrepareDraw(float window_size_ratio);
    void Renderer(const nextfloor::objects::Mesh& mesh);

    std::unique_ptr<nextfloor::objects::Mesh> universe_{nullptr};
    Character* player_{nullptr};
    std::list<Camera*> game_cameras_;
    std::unique_ptr<CollisionEngine> collision_engine_{nullptr};
    RendererFactory* renderer_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_GAMELEVEL_H_
