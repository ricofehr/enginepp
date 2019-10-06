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
#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class DemoLevel
 *  @brief Define a determinitic demo level
 */
class DemoLevel : public Level {

public:
    DemoLevel(const nextfloor::objects::MeshFactory& factory);
    ~DemoLevel() final = default;

    DemoLevel(DemoLevel&&) = default;
    DemoLevel& operator=(DemoLevel&&) = default;
    DemoLevel(const DemoLevel&) = delete;
    DemoLevel& operator=(const DemoLevel&) = delete;

    void toready() final { universe_->toready(); }
    void UpdateCameraOrientation(HIDPointer angles, float input_fov) final;
    void ExecutePlayerAction(Action* command, double elapsed_time) final;
    void Move() final;
    void Draw(nextfloor::renderer::RendererFactory* renderer_factory) final;

private:
    void GenerateUniverse(const nextfloor::objects::MeshFactory& factory);
    void SetActiveCamera(nextfloor::objects::Camera* active_camera);

    std::unique_ptr<nextfloor::objects::Mesh> universe_{nullptr};
    nextfloor::objects::Mesh* player_{nullptr};
    std::list<nextfloor::objects::Camera*> game_cameras_;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_DEMOLEVEL_H_
