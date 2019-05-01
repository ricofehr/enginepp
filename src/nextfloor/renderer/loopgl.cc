/**
 *  @file loopgl.cc
 *  @brief LoopGL class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/loopgl.h"

#include <fstream>
#include <iostream>

#include "nextfloor/universe/dynamic/camera.h"
#include "nextfloor/hid/input_handler.h"
#include "nextfloor/graphics/shape3d.h"
#include "nextfloor/core/config_engine.h"
#include "nextfloor/core/global_timer.h"

namespace nextfloor {

namespace renderer {

/* Define LoopGL global static class variables */
GLuint LoopGL::sProgramId = -1;
GLuint LoopGL::sMatrixId = -1;

namespace {

static nextfloor::universe::Universe* sUniverse = nullptr;
static GLFWwindow* sGLWindow = nullptr;

/**
 *   Compile and Load shader from files to ram
 *   2 shaders are compiled: vertex and fragment shaders.
 */
void LoadShaders()
{
    using nextfloor::core::ConfigEngine;

    const char* vertex_file_path = "glsl/SimpleVertexShader.vertexshader";
    const char* fragment_file_path = "glsl/SimpleFragmentShader.fragmentshader";

    /* Create the shaders */
    GLuint vertexshader_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentshader_id = glCreateShader(GL_FRAGMENT_SHADER);

    /* Read the Vertex Shader code from the file */
    std::string vertexshader_code;
    std::ifstream vertexshader_stream(vertex_file_path, std::ios::in);
    if (vertexshader_stream.is_open()) {
        std::string line = "";
        while(getline(vertexshader_stream, line)) {
            vertexshader_code += "\n" + line;
        }
        vertexshader_stream.close();
    } else {
        std::cerr << "Impossible to open " << vertex_file_path << std::endl;
        getchar();
        return;
    }

    /* Read the Fragment Shader code from the file */
    std::string fragmentshader_code;
    std::ifstream fragmentshader_stream(fragment_file_path, std::ios::in);
    if(fragmentshader_stream.is_open()){
        std::string line = "";
        while(getline(fragmentshader_stream, line)) {
            fragmentshader_code += "\n" + line;
        }
        fragmentshader_stream.close();
    }

    GLint result = GL_FALSE;
    int info_log_length;

    /* Compile Vertex Shader */
    if (ConfigEngine::getSetting<int>("debug") > ConfigEngine::kDEBUG_TEST) {
        std::cout << "Compiling shader : " << vertex_file_path << std::endl;
    }
    const char* vertexsource_pointer = vertexshader_code.c_str();
    glShaderSource(vertexshader_id, 1, &vertexsource_pointer , nullptr);
    glCompileShader(vertexshader_id);

    /* Check Vertex Shader */
    glGetShaderiv(vertexshader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertexshader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> vertexshader_error_message(info_log_length + 1);
        glGetShaderInfoLog(vertexshader_id, info_log_length, nullptr, &vertexshader_error_message[0]);
        std::cerr << &vertexshader_error_message[0];
    }

    /* Compile Fragment Shader */
    if (ConfigEngine::getSetting<int>("debug") > ConfigEngine::kDEBUG_TEST) {
        std::cout << "Compiling shader : " << fragment_file_path << std::endl;
    }
    const char* fragmentsource_pointer = fragmentshader_code.c_str();
    glShaderSource(fragmentshader_id, 1, &fragmentsource_pointer, nullptr);
    glCompileShader(fragmentshader_id);

    /* Check Fragment Shader */
    glGetShaderiv(fragmentshader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragmentshader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> fragmentshader_error_message(info_log_length + 1);
        glGetShaderInfoLog(fragmentshader_id, info_log_length, nullptr, &fragmentshader_error_message[0]);
        std::cerr << &fragmentshader_error_message[0];
    }

    /* Link the program */
    if (ConfigEngine::getSetting<int>("debug") > ConfigEngine::kDEBUG_TEST) {
        std::cout << "Linking program" << std::endl;
    }
    LoopGL::sProgramId = glCreateProgram();
    glAttachShader(LoopGL::sProgramId, vertexshader_id);
    glAttachShader(LoopGL::sProgramId, fragmentshader_id);
    glLinkProgram(LoopGL::sProgramId);

    /* Check the program */
    glGetProgramiv(LoopGL::sProgramId, GL_LINK_STATUS, &result);
    glGetProgramiv(LoopGL::sProgramId, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(LoopGL::sProgramId, info_log_length, nullptr, &program_error_message[0]);
        std::cerr << &program_error_message[0];
    }

    glDetachShader(LoopGL::sProgramId, vertexshader_id);
    glDetachShader(LoopGL::sProgramId, fragmentshader_id);

    glDeleteShader(vertexshader_id);
    glDeleteShader(fragmentshader_id);
}

/**
 *    Draw Current Universe
 */
void Draw()
{
    using nextfloor::core::ConfigEngine;

    /* Enable depth test */
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(LoopGL::sProgramId);

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

void LoopGL::InitGL()
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

void LoopGL::Loop(nextfloor::universe::Universe* uni)
{
    using nextfloor::core::ConfigEngine;
    using nextfloor::core::GlobalTimer;
    using nextfloor::universe::commands::Command;
    using nextfloor::universe::dynamic::Camera;

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
    LoopGL::sMatrixId = glGetUniformLocation(LoopGL::sProgramId, "MVP");

    /* Ensure prerequisite */
    assert(sUniverse != nullptr);
    assert(LoopGL::sMatrixId != -1);
    assert(LoopGL::sProgramId != -1);

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
