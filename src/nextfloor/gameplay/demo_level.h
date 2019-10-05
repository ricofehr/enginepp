/**
 *  @file demo_level.h
 *  @brief Demo Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_
#define NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_

#include "nextfloor/gameplay/level.h"

#include <memory>

#include "nextfloor/objects/mesh.h"


namespace nextfloor {

namespace gameplay {

/**
 *  @class DemoLevel
 *  @brief Define a determinitic demo level
 */
class DemoLevel : public Level {

public:
    DemoLevel();
    ~DemoLevel() final = default;

    DemoLevel(DemoLevel&&) = default;
    DemoLevel& operator=(DemoLevel&&) = default;
    DemoLevel(const DemoLevel&) = delete;
    DemoLevel& operator=(const DemoLevel&) = delete;

    void toready() final { universe_->toready(); }
    void UpdateCameraOrientation(HIDPointer angles, float input_fov) final;
    void ExecutePlayerAction(Action* command, double elapsed_time) final;
    void Move() final;
    void Draw() final;

private:
    void GenerateUniverse();
    void SetActiveCamera(nextfloor::objects::Camera* active_camera);

    std::unique_ptr<nextfloor::objects::Mesh> universe_{nullptr};
    nextfloor::objects::Mesh* player_{nullptr};
    std::list<nextfloor::objects::Camera*> game_cameras_;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_
