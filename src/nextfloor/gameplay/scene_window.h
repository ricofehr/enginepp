/**
 *  @file scene_window.h
 *  @brief SceneWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_SCENEWINDOW_H_
#define NEXTFLOOR_GAMEPLAY_SCENEWINDOW_H_

namespace nextfloor {

namespace gameplay {

class SceneWindow {

public:
    virtual ~SceneWindow() = default;

    virtual void PrepareDisplay() = 0;
    virtual void SwapBuffers() = 0;

    virtual void* window() const = 0;

    virtual float getWindowRatio() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_SCENEWINDOW_H_
