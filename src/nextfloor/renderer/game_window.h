/**
 *  @file game_window.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GAMEWINDOW_H_
#define NEXTFLOOR_RENDERER_GAMEWINDOW_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

#include "nextfloor/universe/model3d.h"
#include "nextfloor/universe/dynamic/camera.h"
#include "nextfloor/renderer/fragment_shader.h"
#include "nextfloor/renderer/vertex_shader.h"

namespace nextfloor {

namespace renderer {

class GameWindow {

public:

    GameWindow();

    GameWindow(GameWindow&&) = default;

    GameWindow& operator=(GameWindow&&) = default;

    GameWindow(const GameWindow&) = default;

    GameWindow& operator=(const GameWindow&) = default;

    ~GameWindow();

    void Initialization();
    void PrepareDisplay();
    void SetCamera(nextfloor::universe::dynamic::Camera* camera);
    void SwapBuffers();
    void UpdateMoveFactor();

    /**
     *  Accessors
     */
    GLFWwindow* glfw_window() { return glfw_window_; }

    /**
     *  GameWindow Global Variables Accessors
     */
    static nextfloor::universe::dynamic::Camera*  getCamera()
    {
        using nextfloor::universe::dynamic::Camera;
        return (Camera*) camera_;
    }
    static GLuint getMatrixId() { return matrix_id_; }
    static GLuint getProgramId() { return program_id_; }
    static float getWidth() { return window_width_; }
    static float getHeight() { return window_height_; }
    static float getMoveFactor() { return move_factor_; }

private:

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

    std::unique_ptr<FragmentShader> fragment_shader_{nullptr};
    std::unique_ptr<VertexShader> vertex_shader_{nullptr};

    /** Fps target for speed movement compute on Engine */
    static constexpr float kFpsBase = 60.0f;

    static nextfloor::universe::Model3D* camera_;
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

#endif // NEXTFLOOR_RENDERER_GAMEWINDOW_H_