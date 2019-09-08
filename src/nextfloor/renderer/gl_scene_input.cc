/**
 *  @file gl_scene_input.cc
 *  @brief GlSceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_scene_input.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

GlSceneInput::GlSceneInput(SceneWindow* window)
{
    glfw_window_ = static_cast<GLFWwindow*>(window->window());
    glfwSetInputMode(glfw_window_, GLFW_STICKY_KEYS, GL_TRUE);
}

void GlSceneInput::PollEvents()
{
    glfwPollEvents();
}

bool GlSceneInput::IsCloseWindowEventOccurs()
{
    return glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwWindowShouldClose(glfw_window_);
}

bool GlSceneInput::IsPressed(int ACTION_BUTTON)
{
    return glfwGetKey(glfw_window_, getKeyValueForAction(ACTION_BUTTON)) == GLFW_PRESS;
}

/**
 * Map Buttun with GL Constants
 * @param  ACTION_BUTTON Button Identifiant
 * @return               GL Constant value
 */
int GlSceneInput::getKeyValueForAction(int ACTION_BUTTON)
{
    switch (ACTION_BUTTON) {
    case kINPUT_UP: return GLFW_KEY_UP;
    case kINPUT_DOWN: return GLFW_KEY_DOWN;
    case kINPUT_LEFT: return GLFW_KEY_LEFT;
    case kINPUT_RIGHT: return GLFW_KEY_RIGHT;
    case kINPUT_JUMP: return GLFW_KEY_SPACE;
    case kINPUT_FIRE: return GLFW_KEY_LEFT_CONTROL;
    case kINPUT_RUN: return GLFW_KEY_LEFT_SHIFT;
    }

    return 0;
}


glm::vec2 GlSceneInput::GetCursorPos()
{
    /* Get mouse position */
    double xpos = 0, ypos = 0;
    glfwGetCursorPos(glfw_window_, &xpos, &ypos);
    return glm::vec2(xpos, ypos);
}

void GlSceneInput::SetCursorPos(float x, float y)
{
    glfwSetCursorPos(glfw_window_, x, y);
}

// void GlSceneInput::SetScrollCallBack(void (*on_scroll)(void* window, double delta_x, double delta_y))
// {
//     glfwSetScrollCallback(glfw_window_, on_scroll);
// }

}  // namespace renderer

}  // namespace nextfloor
