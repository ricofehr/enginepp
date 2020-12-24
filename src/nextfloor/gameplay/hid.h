/**
 *  @file hid.h
 *  @brief HID (Human Interface Device) abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_HID_H_
#define NEXTFLOOR_GAMEPLAY_HID_H_

namespace nextfloor {

namespace gameplay {


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

    virtual bool isPressed(int action_button) = 0;
    virtual bool isReleased(int action_button) = 0;

    /**
     *  Get Current Delta angles for HID pointer
     *  @param  elapsed_time since last record
     *  @return        HID Delta pointer angles (horizontal / vertical)
     */
    virtual HIDPointer RecordHIDPointer(double elapsed_time) = 0;

    virtual void PollEvents() = 0;
    virtual void ResetPointer() = 0;

};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_HID_H_