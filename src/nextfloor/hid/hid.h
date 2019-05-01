/**
 *  @file hid.h
 *  @brief HID (Human Interface Device) abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_HID_H_
#define NEXTFLOOR_HID_HID_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

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
 *  Define delta angles for the hid pointer
 */
typedef struct {
    double horizontal_delta_angle;
    double vertical_delta_angle;
} HIDPointer;

/**
 *  HID Interface class, define design for all devices
 */
class HID {

public:

    /**
     *  Default Move constructor
     */
    HID(HID&&) = default;

    /**
     *  Default Move assignment
     */
    HID& operator=(HID&&) = default;

    /**
     *  Default Copy constructor
     */
    HID(const HID&) = default;

    /**
     *  Default Copy assignment
     */
    HID& operator=(const HID&) = default;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~HID() = default;

    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    virtual bool isPressed(GLFWwindow* window, int ACTION_BUTTON) = 0;

    /**
     *  Get Current Delta angles for HID pointer
     *  @param  window  GL Main Window
     *  @return        HID Delta pointer angles (horizontal / vertical)
     */
    virtual HIDPointer RecordHIDPointer(GLFWwindow* window) = 0;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    virtual float RecordFOV(GLFWwindow* window) = 0;


    /**
     *  Action Button
     */
    static constexpr int kHID_UP = 0;
    static constexpr int kHID_DOWN = 1;
    static constexpr int kHID_LEFT = 2;
    static constexpr int kHID_RIGHT = 3;
    static constexpr int kHID_JUMP = 4;
    static constexpr int kHID_FIRE = 5;
    static constexpr int kHID_RUN = 6;

    static constexpr int kACTIONS = 7;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    HID() = default;

};

} // namespace hid

} // namespace nextfloor

#endif // NEXTFLOOR_HID_HID_H_