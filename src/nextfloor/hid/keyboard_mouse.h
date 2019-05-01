/**
 *  @file keyboard_mouse.h
 *  @brief KeyboardMouse Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_KEYBOARDMOUSE_H_
#define NEXTFLOOR_HID_KEYBOARDMOUSE_H_

#include "nextfloor/hid/hid.h"

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
 *  Manages keyboard / Mouse Controllers
 *  Implements HID Interface
 */
class KeyboardMouse : public HID {

public:
    /**
     *  Default constructor
     */
    KeyboardMouse() = default;

    /**
     *  Default Move constructor
     */
    KeyboardMouse(KeyboardMouse&&) = default;

    /**
     *  Default Move assignment
     */
    KeyboardMouse& operator=(KeyboardMouse&&) = default;

    /**
     *  Default Copy constructor
     */
    KeyboardMouse(const KeyboardMouse&) = default;

    /**
     *  Default Copy assignment
     */
    KeyboardMouse& operator=(const KeyboardMouse&) = default;

    /**
     *  Default destructor
     */
    ~KeyboardMouse() override = default;

    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    bool isPressed(GLFWwindow* window, int ACTION_BUTTON) override final;

    /**
     * Get HID Pointer angles changes
     * @param  window GL Window
     * @return        HIDPointer struct with orientation angles changes
     */
    HIDPointer RecordHIDPointer(GLFWwindow* window) override final;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    float RecordFOV(GLFWwindow* window) override final;
};

} // namespace hid

} // namespace nextfloor

#endif // NEXTFLOOR_HID_KEYBOARDMOUSE_H_