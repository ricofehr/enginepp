/**
 *  @file gl_scene_window.cc
 *  @brief GlSceneWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_scene_window.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cassert>

#include "nextfloor/core/common_services.h"
#include "nextfloor/core/config_parser.h"

namespace nextfloor {

namespace renderer {

namespace {

static bool sInstanciated = false;

void InitGLFW()
{
    if (!glfwInit()) {
        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine("Failed to initialize GLFW");
        CommonServices::getExit()->ExitOnError();
    }
}

void ConfigGL()
{
    glfwWindowHint(GLFW_SAMPLES, 4); /* 4x antialiasing */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);
}

void InitGlew()
{
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine("Failed to initialize GLEW");
        CommonServices::getExit()->ExitOnError();
    }
}

void ClearWindow()
{
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
}

}  // anonymous namespace

GlSceneWindow::GlSceneWindow()
{
    assert(!sInstanciated);
    sInstanciated = true;

    /**
     *  Subroutines Order is matters
     */
    InitGLFW();
    InitWindowSize();
    ConfigGL();
    CreateWindow();
    InitGlew();
    ClearWindow();
    InitRefreshRate();
    InitVSync();
    InitPolygonMode();
    CheckPrerequisites();
}

void GlSceneWindow::InitWindowSize()
{
    nextfloor::core::ConfigParser* config = nextfloor::core::CommonServices::getConfig();
    window_width_ = config->getWindowWidth();
    window_height_ = config->getWindowHeight();
}

void GlSceneWindow::CreateWindow()
{
    /* Open a window and create its OpenGL context (use glfwGetPrimaryMonitor() on third parameter for FS) */
    glfw_window_ = glfwCreateWindow(window_width_, window_height_, "=== Engine ===", nullptr, nullptr);
    if (glfw_window_ == nullptr) {
        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine("Failed to open GLFW window");
        glfwTerminate();
        CommonServices::getExit()->ExitOnError();
    }
    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(glfw_window_);
}

void GlSceneWindow::InitRefreshRate()
{
    GLFWmonitor* gl_monitor = glfwGetPrimaryMonitor();
    assert(gl_monitor != NULL);
    const GLFWvidmode* gl_mode = glfwGetVideoMode(gl_monitor);
    assert(gl_mode != NULL);
    monitor_refresh_rate_ = gl_mode->refreshRate;
}

void GlSceneWindow::InitVSync()
{
    using nextfloor::core::CommonServices;
    is_vsync_enabled_ = CommonServices::getConfig()->isVsync();
    if (!is_vsync_enabled_) {
        glfwSwapInterval(0);
    }
    else {
        glfwSwapInterval(1);
    }
}

void GlSceneWindow::InitPolygonMode()
{
    using nextfloor::core::CommonServices;
    if (CommonServices::getConfig()->isGridMode()) {
        polygon_mode_ = GL_LINE;
    }
    else {
        polygon_mode_ = GL_FILL;
    }
}

void GlSceneWindow::CheckPrerequisites()
{
    assert(glfw_window_ != nullptr);
}

void GlSceneWindow::PrepareDisplay()
{
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    InitPolygonMode();
    glPolygonMode(GL_FRONT_AND_BACK, polygon_mode_);
}

void GlSceneWindow::SwapBuffers()
{
    /* Swap buffers and poll */
    glfwSwapBuffers(glfw_window_);
}

GlSceneWindow::~GlSceneWindow() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace renderer

}  // namespace nextfloor
