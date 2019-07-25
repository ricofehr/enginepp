/**
 *  @file game_window.cc
 *  @brief GameWindow class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/game_window.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

namespace {

static bool sInstanciated = false;

static void InitGLFW()
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        exit(-1);
    }
}

static void ConfigGL()
{
    glfwWindowHint(GLFW_SAMPLES, 4); /* 4x antialiasing */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);
}

static void InitGlew()
{
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        exit(-1);
    }
}

static void ClearWindow()
{
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
}

} // anonymous namespace

/**
 *  GameWindow Global Variables Init
 */
nextfloor::objects::Camera* GameWindow::camera_ = nullptr;
float GameWindow::window_width_ = 1200.0f;
float GameWindow::window_height_ = 740.0f;
GLuint GameWindow::matrix_id_ = -1;
GLuint GameWindow::program_id_ = -1;
float GameWindow::move_factor_ = 1.0f;

GameWindow::GameWindow()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void GameWindow::InitWindowSize()
{
    using nextfloor::core::CommonServices;

    window_width_ = CommonServices::getConfig()->getWindowWidth();
    window_height_ = CommonServices::getConfig()->getWindowHeight();
}

void GameWindow::CreateWindow()
{
    /* Open a window and create its OpenGL context (use glfwGetPrimaryMonitor() on third parameter for FS) */
    glfw_window_ = glfwCreateWindow(window_width_, window_height_,
                                    "=== Engine ===", nullptr, nullptr);
    if(glfw_window_ == nullptr) {
        std::cerr << "Failed to open GLFW window\n";
        glfwTerminate();
        exit(-1);
    }
    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(glfw_window_);
}

void GameWindow::InitRefreshRate()
{
    auto gl_monitor = glfwGetPrimaryMonitor();
    assert(gl_monitor != NULL);
    auto gl_mode = glfwGetVideoMode(gl_monitor);
    assert(gl_mode != NULL);
    monitor_refresh_rate_ = gl_mode->refreshRate;
}

void GameWindow::InitVAO()
{
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
}

void GameWindow::InitVSync()
{
    using nextfloor::core::CommonServices;

    is_vsync_enabled_ = CommonServices::getConfig()->isVsync();
    if (!is_vsync_enabled_) {
        glfwSwapInterval(0);
    }
}

void GameWindow::InitProgramId()
{
    program_id_ = glCreateProgram();
}

void GameWindow::LoadShaders()
{
    using nextfloor::core::CommonServices;
    vertex_shader_ = CommonServices::getFactory()->MakeVertexShader(kVERTEXFILEPATH);
    fragment_shader_ = CommonServices::getFactory()->MakeFragmentShader(kFRAGMENTFILEPATH);

    vertex_shader_->LoadShader();
    fragment_shader_->LoadShader();
    vertex_shader_->LinkShader();
    fragment_shader_->LinkShader();
    Shader::CheckProgram();
    vertex_shader_->DetachShader();
    fragment_shader_->DetachShader();
}

void GameWindow::InitMatrixId()
{
    matrix_id_ = glGetUniformLocation(program_id_, "MVP");
}

void GameWindow::InitPolygonMode()
{
    using nextfloor::core::CommonServices;

    if (CommonServices::getConfig()->isGridMode()) {
        polygon_mode_ = GL_LINE;
    } else {
        polygon_mode_ = GL_FILL;
    }
}

void GameWindow::CheckPrerequisites()
{
    assert(glfw_window_ != nullptr);
    assert(matrix_id_ != -1);
    assert(program_id_ != -1);
}

/**
 *  Subroutines Order is matters
 */
void GameWindow::Initialization()
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

void GameWindow::PrepareDisplay()
{
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(program_id_);

    glPolygonMode(GL_FRONT_AND_BACK, polygon_mode_);

    /** The scene drawn needs a camera */
    assert(camera_ != nullptr);
}

void GameWindow::UpdateMoveFactor()
{
    using nextfloor::core::CommonServices;

    /** Fps displayed can't be greater than monitor refresh rate */
    auto fps_real = CommonServices::getTimer()->getLoopCountBySecond();
    if (fps_real > monitor_refresh_rate_) {
        fps_real = monitor_refresh_rate_;
    }

    move_factor_ = kFpsBase / fps_real;
}

void GameWindow::SetCamera(nextfloor::objects::Camera* camera)
{
    camera_ = camera;
}

void GameWindow::SwapBuffers()
{
    /* Swap buffers and poll */
    glfwSwapBuffers(glfw_window_);
}

GameWindow::~GameWindow()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace renderer

} // namespace nextfloor
