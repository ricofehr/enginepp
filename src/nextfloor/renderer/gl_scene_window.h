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
    GlSceneWindow(ShaderFactory* shader_factory);
    ~GlSceneWindow() noexcept final;

    void PrepareDisplay() final;
    void SwapBuffers() final;

    void* window() const final { return glfw_window_; }
    GLuint getMatrixId() const final { return matrix_id_; }
    GLuint getProgramId() const final { return program_id_; }
    float getWindowRatio() const final { return window_width_ / window_height_; }

private:
    static constexpr const char kVertexFilePath[] = "glsl/SimpleVertexShader.vertexshader";
    static constexpr const char kFragmentFilePath[] = "glsl/SimpleFragmentShader.fragmentshader";

    void LoadShaders();
    void InitVAO();
    void InitVSync();
    void CreateWindow();
    void InitWindowSize();
    void InitRefreshRate();
    void InitPolygonMode();
    void InitProgramId();
    void InitMatrixId();
    void CheckPrerequisites();

    GLFWwindow* glfw_window_{nullptr};
    GLuint vao_;
    bool is_vsync_enabled_{true};
    GLuint polygon_mode_{GL_LINE};
    int monitor_refresh_rate_{0};
    GLuint matrix_id_{0};
    GLuint program_id_{0};

    Shader* fragment_shader_{nullptr};
    Shader* vertex_shader_{nullptr};
    ShaderFactory* shader_factory_;

    float window_width_{1200.0f};
    float window_height_{800.0f};
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_
