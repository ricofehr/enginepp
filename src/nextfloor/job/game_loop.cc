/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/job/game_loop.h"

#include "nextfloor/renderer/fragment_shader.h"
#include "nextfloor/renderer/vertex_shader.h"

#include "nextfloor/universe/dynamic/camera.h"
#include "nextfloor/hid/input_handler.h"
#include "nextfloor/graphics/shape3d.h"
#include "nextfloor/core/config_engine.h"
#include "nextfloor/core/global_timer.h"

namespace nextfloor {

namespace job {

/* Define GameLoop global static class variables */
GLuint GameLoop::sMatrixId = -1;

namespace {

static nextfloor::universe::Universe* sUniverse = nullptr;
static GLFWwindow* sGLWindow = nullptr;

/**
 *   Compile and Load shader from files to ram
 *   2 shaders are compiled: vertex and fragment shaders.
 */
void LoadShaders()
{
    using nextfloor::renderer::Shader;
    using nextfloor::renderer::VertexShader;
    using nextfloor::renderer::FragmentShader;

    const char* vertex_file_path = "glsl/SimpleVertexShader.vertexshader";
    const char* fragment_file_path = "glsl/SimpleFragmentShader.fragmentshader";

    VertexShader vertex_shader(vertex_file_path);
    FragmentShader fragment_shader(fragment_file_path);

    vertex_shader.LoadShader();
    fragment_shader.LoadShader();
    vertex_shader.LinkShader();
    fragment_shader.LinkShader();
    Shader::CheckProgram();
    vertex_shader.DetachShader();
    fragment_shader.DetachShader();
}

/**
 *    Draw Current Universe
 */
void Draw()
{
    using nextfloor::core::ConfigEngine;
    using nextfloor::renderer::Shader;

    /* Enable depth test */
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(Shader::sProgramId);

    /* Fill polygon */
    if (ConfigEngine::getSetting<bool>("grid")) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    sUniverse->Draw();

    /* Swap buffers and poll */
    glfwSwapBuffers(sGLWindow);
}

/**
 *   Compute current fps and display it (if debug enabled)
 */
int Fps()
{
    using nextfloor::core::ConfigEngine;
    using nextfloor::core::GlobalTimer;

    static double sLastTime = GlobalTimer::sTotalTime;
    static int sNbFrames = 0;
    static bool sFirstLoop = true;

    int ret = 0;

    sNbFrames++;
    if (GlobalTimer::sTotalTime - sLastTime >= 1.0) {
        int debug = ConfigEngine::getSetting<int>("debug");

        /* Header for test datas output */
        if (sFirstLoop &&
            debug == ConfigEngine::kDEBUG_TEST) {
            std::cout << "TIME:FPS:NBOBJALL:NBOBJMOVE" << std::endl;
        }
        /* Print if debug */
        if (debug == ConfigEngine::kDEBUG_ALL) {
            std::cout << 1000.0 / static_cast<double>(sNbFrames) << " ms/frame - ";
        }

        if (debug == ConfigEngine::kDEBUG_PERF || debug == ConfigEngine::kDEBUG_ALL) {
            std::cout << sNbFrames << " fps - ";
            std::cout << sUniverse->countRoomsChilds(false) << " objects ("
                      << sUniverse->countRoomsChilds(true) << " displayed) in ";
            std::cout << sUniverse->countRooms(false) << " rooms (" << sUniverse->countRooms(true) << " displayed)";
            std::cout << std::endl;
        }

        /* Update movefactor for objects */
        nextfloor::graphics::Shape3D::sMoveFactor = 60.0f / sNbFrames;
        sUniverse->toready();

        /* Test datas output */
        if (debug == ConfigEngine::kDEBUG_TEST) {
            std::cout << static_cast<int>(GlobalTimer::sTotalTime - sLastTime) << ":" << sNbFrames << ":";
            std::cout << sUniverse->countRoomsChilds(true) << ":"
                      << sUniverse->countRoomsMovingChilds(true) << std::endl;
        }

        /* Reset timer */
        ret = sNbFrames;
        sNbFrames = 0;
        sLastTime += 1.0;
        sFirstLoop = false;
    }

    int end_time = ConfigEngine::getSetting<int>("execution_time");
    if (end_time && GlobalTimer::sTotalTime >= end_time) {
        exit(0);
    }

    return ret;
}

} // anonymous namespace

void GameLoop::InitGL()
{
    using nextfloor::core::ConfigEngine;

    /* Default value for width and height */
    float window_width = 1200.0f;
    float window_height = 740.0f;

    /* Initialise GLFW */
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        exit(-1);
    }

    /* Check width and height into config file and ensure values are setted */
    window_width = ConfigEngine::getSetting<float>("width");
    window_height = ConfigEngine::getSetting<float>("height");

    glfwWindowHint(GLFW_SAMPLES, 4); /* 4x antialiasing */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); /* OpenGL 4.1 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);

    /* Open a window and create its OpenGL context (use glfwGetPrimaryMonitor() on third parameter for FS) */
    sGLWindow = glfwCreateWindow(window_width, window_height, "=== Engine ===", nullptr, nullptr);
    if(sGLWindow == nullptr) {
        std::cerr << "Failed to open GLFW window\n";
        glfwTerminate();
        exit(-1);
    }
    glfwSetInputMode(sGLWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(sGLWindow);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        exit(-1);
    }

    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    /* VAO */
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
}

void GameLoop::Loop(nextfloor::universe::Universe* uni)
{
    using nextfloor::core::ConfigEngine;
    using nextfloor::core::GlobalTimer;
    using nextfloor::universe::commands::Command;
    using nextfloor::universe::dynamic::Camera;
    using nextfloor::renderer::Shader;

    /* Init Global universe var */
    sUniverse = uni;

    /* Creates InputHanedler object */
    nextfloor::hid::InputHandler input_handler(sGLWindow);

    /* Vsync Setting (default is enable) */
    if (!ConfigEngine::getSetting<bool>("vsync")) {
        glfwSwapInterval(0);
    }

    /* Create and compile our GLSL program from the shader */
    LoadShaders();

    /* Get a handle for our "MVP" uniform */
    GameLoop::sMatrixId = glGetUniformLocation(Shader::sProgramId, "MVP");

    /* Ensure prerequisite */
    assert(sUniverse != nullptr);
    assert(GameLoop::sMatrixId != -1);
    assert(Shader::sProgramId != -1);

    int nb_frames = 0;
    bool is_draw = true;
    bool is_released = true;
    nextfloor::universe::Model3D* camera = sUniverse->get_camera();

    /* Draw if window is focused and destroy window if ESC is pressed */
    do {
        GlobalTimer::LoopTimer();
        ((Camera *)camera)->ComputeFOV(input_handler.RecordFOV());
        ((Camera *)camera)->ComputeOrientation(input_handler.RecordHIDPointer());
        Command* command = input_handler.HandlerInput();
        if (command) {
            command->execute(camera);
        }

        /* Pause button */
        /*
        if (glfwGetKey(sGLWindow, GLFW_KEY_P) == GLFW_PRESS) {
            if (is_released) {
                is_draw = is_draw ? false : true;
            }
            is_released = false;
        } else {
            is_released = true;
        }*/

        if (is_draw) {
            Draw();
            Fps();
        }

        glfwPollEvents();
    }
    while (glfwGetKey(sGLWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
           && glfwWindowShouldClose(sGLWindow) == 0);
}

} // namespace renderer

} // namespace nextfloor
