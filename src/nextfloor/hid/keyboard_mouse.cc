/**
 *  @file keyboard_mouse.cc
 *  @brief KeyboardMouse Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/keyboard_mouse.h"

#include "nextfloor/core/common_services.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace hid
 *  @brief hid namespace
 */
namespace hid {

namespace {

/* Start record camera position after 10 drawing scene */
static int sSkipTime = 0;

/* Record mouse wheel scroll */
static float sScrollY = 0.0f;

/**
 * Map Buttun with GL Constants
 * @param  ACTION_BUTTON Button Identifiant
 * @return               GL Constant value
 */
int getKeyValueForAction(int ACTION_BUTTON)
{
    switch(ACTION_BUTTON) {
        case HID::kHID_UP:
            return GLFW_KEY_UP;
        case HID::kHID_DOWN:
            return GLFW_KEY_DOWN;
        case HID::kHID_LEFT:
            return GLFW_KEY_LEFT;
        case HID::kHID_RIGHT:
            return GLFW_KEY_RIGHT;
        case HID::kHID_JUMP:
            return GLFW_KEY_SPACE;
        case HID::kHID_FIRE:
            return GLFW_KEY_LEFT_CONTROL;
        case HID::kHID_RUN:
            return GLFW_KEY_LEFT_SHIFT;
    }

    return 0;
}

/*
 *  OnScroll - callback function who record wheel change
 */
static void OnScroll(GLFWwindow* window, double delta_x, double delta_y)
{
    sScrollY += delta_y;
}

}

/**
 *  Check if a button is pressed
 *  @param  window        GL Main Window
 *  @param  ACTION_BUTTON Button to be checked
 *  @return               True if pressed
 */
bool KeyboardMouse::isPressed(GLFWwindow* window, int ACTION_BUTTON)
{
    return glfwGetKey(window, getKeyValueForAction(ACTION_BUTTON)) == GLFW_PRESS;
}


/**
 * Get HID Pointer angles changes
 * @param  window GL Window
 * @return        HIDPointer struct with orientation angles changes
 */
HIDPointer KeyboardMouse::RecordHIDPointer(GLFWwindow* window)
{
    using nextfloor::core::CommonServices;
    float window_width = CommonServices::getConfig().getSetting<float>("width");
    float window_height = CommonServices::getConfig().getSetting<float>("width");

    const float hid_speed = 0.1f;

    /* Get mouse position */
    double xpos = 0, ypos = 0;

    /* Ensure cursor is well centered before record move */
    if (sSkipTime++ < 10) {
        glfwSetCursorPos(window, window_width/2 , window_height/2);
        xpos = window_width/2;
        ypos = window_height/2;
    } else {
        glfwGetCursorPos(window, &xpos, &ypos);
    }


    HIDPointer pointer = { hid_speed * CommonServices::getTimer().getDeltaTimeSinceLastLoop() * static_cast<float>(window_width/2 - xpos),
                           hid_speed * CommonServices::getTimer().getDeltaTimeSinceLastLoop() * static_cast<float>(window_height/2 - ypos) };

   /* Reset Cursor position at center of screen */
    glfwSetCursorPos(window, window_width/2, window_height/2);

    return pointer;
}

/**
 *  Return current "Field of View"
 *  @param  window GL Main Window
 *  @return        Current Field of View
 */
float KeyboardMouse::RecordFOV(GLFWwindow* window)
{
    const float zoom_sensitivity = -0.2f;
    float delta_fov = 0.0f;

    /* Manage Field of View with mouse wheel */
    glfwSetScrollCallback(window, OnScroll);
    delta_fov = zoom_sensitivity * sScrollY;
    sScrollY = 0.0f;

    return delta_fov;
}

} // namespace hid

} // namespace nextfloor