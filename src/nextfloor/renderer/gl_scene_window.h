/**
 *  @file gl_scene_window.h
 *  @brief GL Scene Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_
#define NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_

#include "nextfloor/renderer/scene_window.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/renderer/shader.h"

namespace nextfloor {

namespace renderer {

class GlSceneWindow : public SceneWindow {

public:
    GlSceneWindow();

    GlSceneWindow(GlSceneWindow&&) = default;
    GlSceneWindow& operator=(GlSceneWindow&&) = default;
    GlSceneWindow(const GlSceneWindow&) = default;
    GlSceneWindow& operator=(const GlSceneWindow&) = default;

    virtual ~GlSceneWindow() override;

    virtual void Initialization() override;
    virtual void PrepareDisplay() override;
    virtual void SwapBuffers() override;
    virtual void UpdateMoveFactor() override;

    virtual void* window() override { return glfw_window_; }
    virtual GLuint getMatrixId() override { return matrix_id_; }
    virtual GLuint getProgramId() override { return program_id_; }
    virtual float getWidth() override { return window_width_; }
    virtual float getHeight() override { return window_height_; }
    virtual float getFpsFixMoveFactor() override { return 1.0f; }  // move_factor_; }

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
    GLuint matrix_id_{0};
    GLuint program_id_{0};

    Shader* fragment_shader_{nullptr};
    Shader* vertex_shader_{nullptr};

    float window_width_{1200.0f};
    float window_height_{800.0f};

    /**
     *  Represents the speed factor between fps taken in initial move computes
     *  and fps displayed at screen
     */
    float move_factor_{1.0f};
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSCENEWINDOW_H_
