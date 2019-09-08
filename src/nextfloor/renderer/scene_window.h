/**
 *  @file scene_window.h
 *  @brief SceneWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SCENEWINDOW_H_
#define NEXTFLOOR_RENDERER_SCENEWINDOW_H_

#include "nextfloor/objects/camera.h"

namespace nextfloor {

namespace renderer {

class SceneWindow {

public:
    virtual ~SceneWindow() = default;

    virtual void Initialization() = 0;
    virtual void PrepareDisplay() = 0;
    virtual void SwapBuffers() = 0;
    virtual void UpdateMoveFactor() = 0;

    virtual void* window() = 0;

    /**
     *  GameWindow Global Variables Accessors
     */
    virtual float getWidth() = 0;
    virtual float getHeight() = 0;
    virtual float getFpsFixMoveFactor() = 0;  // move_factor_; }

    virtual unsigned int getMatrixId() = 0;
    virtual unsigned int getProgramId() = 0;

protected:
    SceneWindow() = default;

    SceneWindow(SceneWindow&&) = default;
    SceneWindow& operator=(SceneWindow&&) = default;
    SceneWindow(const SceneWindow&) = default;
    SceneWindow& operator=(const SceneWindow&) = default;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_SCENEWINDOW_H_
