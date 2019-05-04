/**
 *  @file pad.cc
 *  @brief Pad Class File, TODO: Implement PAD Controller
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/hid/pad.h"

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

/**
 *  Check if a button is pressed
 *  @param  window        GL Main Window
 *  @param  ACTION_BUTTON Button to be checked
 *  @return               True if pressed
 */
bool Pad::isPressed(GLFWwindow* window, int ACTION_BUTTON)
{
    return false;
}

/**
 * Get HID Pointer angles changes
 * @param  window GL Window
 * @return        HIDPointer struct with orientation angles changes
 */
HIDPointer Pad::RecordHIDPointer(GLFWwindow* window)
{

    HIDPointer pointer = { 0.0f, 0.0f };
    return pointer;
}

/**
 *  Return current "Field of View"
 *  @param  window GL Main Window
 *  @return        Current Field of View
 */
float Pad::RecordFOV(GLFWwindow* window)
{
    return 0.0f;
}

} // namespace hid

} // namespace nextfloor