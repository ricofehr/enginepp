/**
 *  @file scene_window.h
 *  @brief SceneWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SCENEWINDOW_H_
#define NEXTFLOOR_RENDERER_SCENEWINDOW_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/renderer/shader.h"

namespace nextfloor {

namespace renderer {

class SceneWindow {

public:

    virtual ~SceneWindow() = default;

    virtual void Initialization() = 0;
    virtual void PrepareDisplay() = 0;
    virtual void SetCamera() = 0;
    virtual void SwapBuffers() = 0;
    virtual void UpdateMoveFactor() = 0;

    virtual GLFWwindow* glfw_window() = 0;

    /**
     *  GameWindow Global Variables Accessors
     */
    static nextfloor::objects::Camera* getCamera() { return camera_; }
    static GLuint getMatrixId() { return matrix_id_; }
    static GLuint getProgramId() { return program_id_; }
    static float getWidth() { return window_width_; }
    static float getHeight() { return window_height_; }
    static float getFpsFixMoveFactor() { return move_factor_; }

protected:

    SceneWindow() = default;

    SceneWindow(SceneWindow&&) = default;
    SceneWindow& operator=(SceneWindow&&) = default;
    SceneWindow(const SceneWindow&) = default;
    SceneWindow& operator=(const SceneWindow&) = default;

    static nextfloor::objects::Camera* camera_;
    static float window_width_;
    static float window_height_;
    static GLuint matrix_id_;
    static GLuint program_id_;

    /**
     *  Represents the speed factor between fps taken in initial move computes
     *  and fps displayed at screen
     */
    static float move_factor_;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_SCENEWINDOW_H_
