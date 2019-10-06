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
    virtual void UpdateMoveFactor(int fps) = 0;

    virtual void* window() const = 0;

    /**
     *  GameWindow Global Variables Accessors
     */
    virtual float getWidth() const = 0;
    virtual float getHeight() const = 0;
    virtual float getFpsFixMoveFactor() const = 0;  // move_factor_; }

    virtual unsigned int getMatrixId() const = 0;
    virtual unsigned int getProgramId() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_SCENEWINDOW_H_
