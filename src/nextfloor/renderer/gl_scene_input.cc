/**
 *  @file gl_scene_input.cc
 *  @brief GlSceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_scene_input.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace nextfloor {

namespace renderer {

GlSceneInput::GlSceneInput(GLFWwindow* glfw_window)
{
    glfw_window_ = glfw_window;
    glfwSetInputMode(glfw_window_, GLFW_STICKY_KEYS, GL_TRUE);
}

void GlSceneInput::PollEvents()
{
    glfwPollEvents();
}

bool GlSceneInput::IsCloseWindowEventOccurs()
{
    return glfwWindowShouldClose(glfw_window_);
}

bool GlSceneInput::IsPressed(int action_button)
{
    bool is_pressed = glfwGetKey(glfw_window_, getKeyValueForAction(action_button)) == GLFW_PRESS;
    if (is_pressed) {
        last_key_pressed_ = action_button;
    }
    return is_pressed;
}

bool GlSceneInput::IsReleased(int action_button)
{
    IsPressed(action_button);

    if (last_key_pressed_ == action_button) {
        bool is_released = glfwGetKey(glfw_window_, getKeyValueForAction(action_button)) == GLFW_RELEASE;
        if (is_released) {
            last_key_pressed_ = -10;
            return true;
        }
    }

    return false;
}

/**
 * Map Buttun with GL Constants
 * @param  ACTION_BUTTON Button Identifiant
 * @return               GL Constant value
 */
int GlSceneInput::getKeyValueForAction(int action_button)
{
    switch (action_button) {
    case kInputUp: return GLFW_KEY_UP;
    case kInputDown: return GLFW_KEY_DOWN;
    case kInputLeft: return GLFW_KEY_LEFT;
    case kInputRight: return GLFW_KEY_RIGHT;
    case kInputSpace: return GLFW_KEY_SPACE;
    case kInputOpenMainMenu: return GLFW_KEY_ESCAPE;
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

}  // namespace renderer

}  // namespace nextfloor
