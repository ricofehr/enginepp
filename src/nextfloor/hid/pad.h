/**
 *  @file pad.h
 *  @brief Pad Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_PAD_H_
#define NEXTFLOOR_HID_PAD_H_

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
 *  Manages Pad Controllers
 *  Implements HID Interface
 */
class Pad : public HID {

public:

    Pad() = default;
    /**
     *  Default Move constructor
     */
    Pad(Pad&&) = default;

    /**
     *  Default Move assignment
     */
    Pad& operator=(Pad&&) = default;

    /**
     *  Default Copy constructor
     */
    Pad(const Pad&) = default;

    /**
     *  Default Copy assignment
     */
    Pad& operator=(const Pad&) = default;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~Pad() override = default;

    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    virtual bool isPressed(GLFWwindow* window, int ACTION_BUTTON) override final;

    /**
     * Get HID Pointer angles changes
     * @param  window GL Window
     * @return        HIDPointer struct with orientation angles changes
     */
    virtual HIDPointer RecordHIDPointer(GLFWwindow* window) override final;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    virtual float RecordFOV(GLFWwindow* window) override final;

};

} // namespace hid

} // namespace nextfloor

#endif // NEXTFLOOR_HID_KEYBOARD_H_