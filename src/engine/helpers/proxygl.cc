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

#include "engine/universe/universe.h"

namespace engine {
namespace helpers {
namespace proxygl {

/* Global variables */
GLFWwindow* kGLWindow = nullptr;
GLuint kProgramId = -1;
GLuint kMatrixId = -1;
engine::universe::Camera *kCam = nullptr;

namespace {

static engine::universe::Universe *universe = nullptr;

/**
*   LoadShaders - Compile and Load shader from files to ram
*
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
        while(getline(vertexshader_stream, line))
            vertexshader_code += "\n" + line;
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
        while(getline(fragmentshader_stream, line))
            fragmentshader_code += "\n" + line;
        fragmentshader_stream.close();
    }

    GLint result = GL_FALSE;
    int info_log_length;

    /* Compile Vertex Shader */
    std::cout << "Compiling shader : " << vertex_file_path << std::endl;
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
    std::cout << "Compiling shader : " << fragment_file_path << std::endl;
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
    std::cout << "Linking program" << std::endl;
    kProgramId = glCreateProgram();
    glAttachShader(kProgramId, vertexshader_id);
    glAttachShader(kProgramId, fragmentshader_id);
    glLinkProgram(kProgramId);

    /* Check the program */
    glGetProgramiv(kProgramId, GL_LINK_STATUS, &result);
    glGetProgramiv(kProgramId, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(kProgramId, info_log_length, nullptr, &program_error_message[0]);
        std::cerr << &program_error_message[0];
    }

    glDetachShader(kProgramId, vertexshader_id);
    glDetachShader(kProgramId, fragmentshader_id);

    glDeleteShader(vertexshader_id);
    glDeleteShader(fragmentshader_id);
}

/**
*    draw - Main display function
*
*    Display function triggered when opengl must display again all polygons
*/
void draw()
{
    /* Enable depth test */
    glEnable(GL_DEPTH_TEST);

    /* Accept fragment if it closer to the camera than the former one */
    glDepthFunc(GL_LESS);

    /* Clear the scene */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* Use defined shaders */
    glUseProgram(kProgramId);

    /* Fill polygon */
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    universe->NextHop();

    /* Swap buffers and poll */
    glfwSwapBuffers(kGLWindow);
    glfwPollEvents();
    usleep(50000);
}

}//namespace

/* Init gl Window */
void initGL()
{
    /* Initialise GLFW */
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        exit(-1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4); /* 4x antialiasing */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); /* OpenGL 3.3 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);

    /* Open a window and create its OpenGL context (use glfwGetPrimaryMonitor() on third parameter for FS) */
    kGLWindow = glfwCreateWindow(kWidthWindow, kHeightWindow, "=== Engine ===", nullptr, nullptr);
    if(kGLWindow == nullptr) {
        std::cerr << "Failed to open GLFW window\n";
        glfwTerminate();
        exit(-1);
    }
    glfwSetInputMode(kGLWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(kGLWindow);
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
void settingsGL(engine::universe::Universe *uni)
{
    /* Init Global universe var */
    universe = uni;
    kCam = uni->cam();

    /* Ensure we can capture keys being pressed below */
    glfwSetInputMode(kGLWindow, GLFW_STICKY_KEYS, GL_TRUE);

    /* Create and compile our GLSL program from the shader */
    LoadShaders();

    /* Get a handle for our "MVP" uniform */
    kMatrixId = glGetUniformLocation(kProgramId, "MVP");

    /* Ensure prerequisite */
    assert(universe != nullptr);
    assert(kMatrixId != -1);
    assert(kProgramId != -1);
    assert(kCam != nullptr);

    /* Draw if window is focused and destroy window if ESC is pressed */
    do{
    	draw();
    }
    while(glfwGetKey(kGLWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
          && glfwWindowShouldClose(kGLWindow) == 0);
}

}//namespace proxygl
}//namespace helpers
}//namespace engine
