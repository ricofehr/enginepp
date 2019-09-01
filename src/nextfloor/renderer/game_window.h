/**
 *  @file game_window.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GAMEWINDOW_H_
#define NEXTFLOOR_RENDERER_GAMEWINDOW_H_

#include "nextfloor/renderer/scene_window.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/renderer/shader.h"

namespace nextfloor {

namespace renderer {

class GameWindow : public SceneWindow {

public:
    GameWindow();

    GameWindow(GameWindow&&) = default;
    GameWindow& operator=(GameWindow&&) = default;
    GameWindow(const GameWindow&) = default;
    GameWindow& operator=(const GameWindow&) = default;

    ~GameWindow();

    void Initialization();
    void PrepareDisplay();
    void SetCamera();
    void SwapBuffers();
    void UpdateMoveFactor();

    GLFWwindow* glfw_window() { return glfw_window_; }

private:
    /** Fps target for speed movement compute on Engine */
    static constexpr float kFpsBase = 60.0f;

    static constexpr const char kVERTEXFILEPATH[] = "glsl/SimpleVertexShader.vertexshader";
    static constexpr const char kFRAGMENTFILEPATH[] = "glsl/SimpleFragmentShader.fragmentshader";

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

    Shader* fragment_shader_{nullptr};
    Shader* vertex_shader_{nullptr};
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GAMEWINDOW_H_
