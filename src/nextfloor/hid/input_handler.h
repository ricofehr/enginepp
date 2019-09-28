/**
 *  @file input_handler.h
 *  @brief InputHandler Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_INPUTHANDLER_H_
#define NEXTFLOOR_HID_INPUTHANDLER_H_

#include "nextfloor/hid/hid.h"
#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace hid {

/**
 *  @class InputHandler
 *  @brief InputHandler base class for handle player inputs
 */
class InputHandler {

public:
    virtual ~InputHandler() = default;

    /**
     *  Get Current State Input
     *  @return Command Object
     */
    virtual nextfloor::actions::Action* HandlerInput() = 0;

    /**
     *  Get HID Pointer angles changes
     *  @return HIDPointer struct
     */
    virtual HIDPointer RecordHIDPointer(double elapsed_time) = 0;

    /**
     *  Get FOV (FieldOfView) Current State
     *  @return FOV value
     */
    virtual float RecordFOV() = 0;

    virtual void PollEvents() = 0;
    virtual bool IsCloseWindowEventOccurs() = 0;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_INPUTHANDLER_H_
