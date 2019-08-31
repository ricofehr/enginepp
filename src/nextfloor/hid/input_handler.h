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
    InputHandler(InputHandler&&) noexcept = default;
    InputHandler& operator=(InputHandler&&) noexcept = default;
    InputHandler(const InputHandler&) = default;
    InputHandler& operator=(const InputHandler&) = default;

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
    virtual HIDPointer RecordHIDPointer() = 0;

    /**
     *  Get FOV (FieldOfView) Current State
     *  @return FOV value
     */
    virtual float RecordFOV() = 0;

protected:
    InputHandler() = default;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_INPUTHANDLER_H_
