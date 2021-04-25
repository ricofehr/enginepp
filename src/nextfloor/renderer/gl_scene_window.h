/**
 *  @file gl_scene_window.h
 *  @brief GL Scene Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_
#define NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_

#include "nextfloor/gameplay/scene_window.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "nextfloor/renderer/shader.h"
#include "nextfloor/renderer/shader_factory.h"

namespace nextfloor {

namespace renderer {

class GlSceneWindow : public nextfloor::gameplay::SceneWindow {

public:
    GlSceneWindow();
    ~GlSceneWindow() noexcept final;

    void PrepareDisplay() final;
    void SwapBuffers() final;

    void* window() const final { return glfw_window_; }
    float getWindowRatio() const final { return window_width_ / window_height_; }

private:
    void InitVSync();
    void CreateWindow();
    void InitWindowSize();
    void InitRefreshRate();
    void InitPolygonMode();
    void CheckPrerequisites();

    GLFWwindow* glfw_window_{nullptr};
    bool is_vsync_enabled_{true};
    GLuint polygon_mode_{GL_LINE};
    int monitor_refresh_rate_{0};

    float window_width_{1200.0f};
    float window_height_{800.0f};
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_
