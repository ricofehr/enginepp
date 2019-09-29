/**
 *  @file gl_scene_window.cc
 *  @brief GlSceneWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_scene_window.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

namespace {

static bool sInstanciated = false;

void InitGLFW()
{
    if (!glfwInit()) {
        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine("Failed to initialize GLFW");
        exit(-1);
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
        exit(-1);
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
    Initialization();
}

/**
 *  Subroutines Order is matters
 */
void GlSceneWindow::Initialization()
{
    InitGLFW();
    InitWindowSize();
    ConfigGL();
    CreateWindow();
    InitGlew();
    ClearWindow();
    InitRefreshRate();
    InitVAO();
    InitVSync();
    InitProgramId();
    LoadShaders();
    InitMatrixId();
    InitPolygonMode();
    CheckPrerequisites();
}

void GlSceneWindow::InitWindowSize()
{
    auto config = nextfloor::core::CommonServices::getConfig();
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
        exit(-1);
    }
    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(glfw_window_);
}

void GlSceneWindow::InitRefreshRate()
{
    auto gl_monitor = glfwGetPrimaryMonitor();
    assert(gl_monitor != NULL);
    auto gl_mode = glfwGetVideoMode(gl_monitor);
    assert(gl_mode != NULL);
    monitor_refresh_rate_ = gl_mode->refreshRate;
}

void GlSceneWindow::InitVAO()
{
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
}

void GlSceneWindow::InitVSync()
{
    auto config = nextfloor::core::CommonServices::getConfig();
    is_vsync_enabled_ = config->isVsync();
    if (!is_vsync_enabled_) {
        glfwSwapInterval(0);
    }
}

void GlSceneWindow::InitProgramId()
{
    program_id_ = glCreateProgram();
}

void GlSceneWindow::LoadShaders()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    vertex_shader_ = factory->MakeVertexShader(kVERTEXFILEPATH, program_id_);
    fragment_shader_ = factory->MakeFragmentShader(kFRAGMENTFILEPATH, program_id_);

    vertex_shader_->LoadShader();
    fragment_shader_->LoadShader();
    vertex_shader_->LinkShader();
    fragment_shader_->LinkShader();
    vertex_shader_->CheckProgram();
    fragment_shader_->CheckProgram();
    vertex_shader_->DetachShader();
    fragment_shader_->DetachShader();
}

void GlSceneWindow::InitMatrixId()
{
    matrix_id_ = glGetUniformLocation(program_id_, "MVP");
}

void GlSceneWindow::InitPolygonMode()
{
    auto config = nextfloor::core::CommonServices::getConfig();
    if (config->isGridMode()) {
        polygon_mode_ = GL_LINE;
    }
    else {
        polygon_mode_ = GL_FILL;
    }
}

void GlSceneWindow::CheckPrerequisites()
{
    assert(glfw_window_ != nullptr);
    assert(matrix_id_ != 0);
    assert(program_id_ != 0);
}

void GlSceneWindow::PrepareDisplay()
{
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(program_id_);

    glPolygonMode(GL_FRONT_AND_BACK, polygon_mode_);
}

void GlSceneWindow::UpdateMoveFactor(int fps)
{
    /** Fps displayed can't be greater than monitor refresh rate */
    if (fps > monitor_refresh_rate_) {
        fps = monitor_refresh_rate_;
    }

    move_factor_ = kFpsBase / fps;
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
