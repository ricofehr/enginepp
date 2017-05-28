#include "engine/helpers/proxygl.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string.h>
#include <unistd.h>

#include <fstream>

#include "engine/geometry/shape3d.h"
#include "engine/universe/universe.h"
#include "engine/helpers/proxy_config.h"

namespace engine {
namespace helpers {
namespace proxygl {

/* Global variables */
GLFWwindow* gGLWindow = nullptr;
GLuint gProgramId = -1;
GLuint gMatrixId = -1;

double gBeginTime = 0.0f;

namespace {

static engine::universe::Universe *universe = nullptr;

/**
*   LoadShaders - Compile and Load shader from files to ram
*   Currently, 2 shaders are compiled: vertex and fragment shaders.
*/
void LoadShaders()
{
    const char *vertex_file_path = "glsl/SimpleVertexShader.vertexshader";
    const char *fragment_file_path = "glsl/SimpleFragmentShader.fragmentshader";

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
    if (ProxyConfig::getSetting<int>("debug") > ProxyConfig::kDEBUG_TEST) {
        std::cout << "Compiling shader : " << vertex_file_path << std::endl;
    }
    const char *vertexsource_pointer = vertexshader_code.c_str();
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
    if (ProxyConfig::getSetting<int>("debug") > ProxyConfig::kDEBUG_TEST) {
        std::cout << "Compiling shader : " << fragment_file_path << std::endl;
    }
    const char *fragmentsource_pointer = fragmentshader_code.c_str();
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
    if (ProxyConfig::getSetting<int>("debug") > ProxyConfig::kDEBUG_TEST) {
        std::cout << "Linking program" << std::endl;
    }
    gProgramId = glCreateProgram();
    glAttachShader(gProgramId, vertexshader_id);
    glAttachShader(gProgramId, fragmentshader_id);
    glLinkProgram(gProgramId);

    /* Check the program */
    glGetProgramiv(gProgramId, GL_LINK_STATUS, &result);
    glGetProgramiv(gProgramId, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(gProgramId, info_log_length, nullptr, &program_error_message[0]);
        std::cerr << &program_error_message[0];
    }

    glDetachShader(gProgramId, vertexshader_id);
    glDetachShader(gProgramId, fragmentshader_id);

    glDeleteShader(vertexshader_id);
    glDeleteShader(fragmentshader_id);
}

/**
*    draw - Main display function
*    Display function triggered when opengl must display again all polygons
*/
void Draw()
{
    /* Enable depth test */
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(gProgramId);

    /* Fill polygon */
    if (ProxyConfig::getSetting<bool>("grid")) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    universe->NextHop();

    /* Swap buffers and poll */
    glfwSwapBuffers(gGLWindow);
}

/*
*   Compute current fps and display it
*/
int Fps(double &last_time, int &nb_frames)
{
    int ret = 0;
    /* Measure speed */
    double current_time = glfwGetTime();

    nb_frames++;
    if (current_time - last_time >= 1.0) {
        int debug = ProxyConfig::getSetting<int>("debug");

        /* Header for test datas output */
        if (gBeginTime == last_time &&
            debug == ProxyConfig::kDEBUG_TEST) {
            std::cout << "TIME:FPS:NBOBJALL:NBOBJMOVE" << std::endl;
        }
        /* Print if debug */
        if (debug == ProxyConfig::kDEBUG_ALL) {
            std::cout << 1000.0 / static_cast<double>(nb_frames) << " ms/frame - ";
        }

        if (debug == ProxyConfig::kDEBUG_PERF || debug == ProxyConfig::kDEBUG_ALL) {
            std::cout << nb_frames << " fps - ";
            std::cout << universe->countObjects(false) << " objects (" << universe->countObjects(true) << " displayed) in ";
            std::cout << universe->countRooms(false) << " rooms (" << universe->countRooms(true) << " displayed)";
            std::cout << std::endl;
        }

        /* Update movefactor for objects */
        engine::geometry::Shape3D::sMoveFactor = 60.0f / nb_frames;
        universe->toready();

        /* Test datas output */
        if (debug == ProxyConfig::kDEBUG_TEST) {
            std::cout << static_cast<int>(current_time - gBeginTime) << ":" << nb_frames << ":";
            std::cout << universe->countObjects(true) << ":" << universe->countMovingObjects(true) << std::endl;
        }

        /* Reset timer */
        ret = nb_frames;
        nb_frames = 0;
        last_time += 1.0;
    }

    int end_time = ProxyConfig::getSetting<int>("execution_time");
    if (end_time && current_time - gBeginTime >= end_time) {
        exit(0);
    }

    return ret;
}

}//namespace

/* Init gl Window */
void InitGL()
{
    /* Default value for width and height */
    float window_width = 1200.0f;
    float window_height = 740.0f;

    /* Initialise GLFW */
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        exit(-1);
    }

    /* Check width and height into config file and ensure values are setted */
    window_width = ProxyConfig::getSetting<float>("width");
    window_height = ProxyConfig::getSetting<float>("height");

    glfwWindowHint(GLFW_SAMPLES, 4); /* 4x antialiasing */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); /* OpenGL 3.3 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);

    /* Open a window and create its OpenGL context (use glfwGetPrimaryMonitor() on third parameter for FS) */
    gGLWindow = glfwCreateWindow(window_width, window_height, "=== Engine ===", nullptr, nullptr);
    if(gGLWindow == nullptr) {
        std::cerr << "Failed to open GLFW window\n";
        glfwTerminate();
        exit(-1);
    }
    glfwSetInputMode(gGLWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(gGLWindow);
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

/* Setup gl scene */
void SettingsGL(engine::universe::Universe *uni)
{
    /* Init Global universe var */
    universe = uni;

    /* Ensure we can capture keys being pressed below */
    glfwSetInputMode(gGLWindow, GLFW_STICKY_KEYS, GL_TRUE);

    /* Vsync Setting (default is enable) */
    if (!ProxyConfig::getSetting<bool>("vsync")) {
        glfwSwapInterval(0);
    }

    /* Create and compile our GLSL program from the shader */
    LoadShaders();

    /* Get a handle for our "MVP" uniform */
    gMatrixId = glGetUniformLocation(gProgramId, "MVP");

    /* Ensure prerequisite */
    assert(universe != nullptr);
    assert(gMatrixId != -1);
    assert(gProgramId != -1);

    gBeginTime = glfwGetTime();
    double last_time = gBeginTime;
    int nb_frames = 0;
    bool is_draw = true;
    bool is_released = true;

    /* Draw if window is focused and destroy window if ESC is pressed */
    do {
        /* Pause button */
        if (glfwGetKey(gGLWindow, GLFW_KEY_P) == GLFW_PRESS) {
            if (is_released) {
                is_draw = is_draw ? false : true;
            }
            is_released = false;
        } else {
            is_released = true;
        }

        if (is_draw) {
            Draw();
            Fps(last_time, nb_frames);
        }

        glfwPollEvents();
    }
    while (glfwGetKey(gGLWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
           && glfwWindowShouldClose(gGLWindow) == 0);
}

}//namespace proxygl
}//namespace helpers
}//namespace engine
