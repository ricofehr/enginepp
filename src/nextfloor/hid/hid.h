/**
 *  @file hid.h
 *  @brief HID (Human Interface Device) abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_HID_H_
#define NEXTFLOOR_HID_HID_H_

namespace nextfloor {

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
    virtual ~HID() = default;

    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    virtual bool isPressed(int ACTION_BUTTON) = 0;

    /**
     *  Get Current Delta angles for HID pointer
     *  @param  window  GL Main Window
     *  @return        HID Delta pointer angles (horizontal / vertical)
     */
    virtual HIDPointer RecordHIDPointer() = 0;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    virtual float RecordFOV() = 0;

    virtual void PollEvents() = 0;

    virtual bool IsCloseWindowEventOccurs() = 0;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_HID_H_