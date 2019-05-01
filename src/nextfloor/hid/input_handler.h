/**
 *  @file input_handler.h
 *  @brief InputHandler Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_INPUTHANDLER_H_
#define NEXTFLOOR_HID_INPUTHANDLER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "nextfloor/hid/hid.h"
#include "nextfloor/universe/commands/command.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::core
 *  @brief Core system
 */
namespace hid {

/**
 *  @class InputHandler
 *  @brief InputHandler maintains input commands from engine
 *  Implement Commands Pattern which ensure loose coopling between action and actors
 */
class InputHandler {

public:

    /**
     *  Constructor
     *  @param  window  GL Main Window
     */
    InputHandler(GLFWwindow *window);

    /**
     *  Default Move constructor
     */
    InputHandler(InputHandler&&) = default;

    /**
     *  Default Move assignment
     */
    InputHandler& operator=(InputHandler&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    InputHandler(const InputHandler&) = default;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    InputHandler& operator=(const InputHandler&) = default;

    /**
     *  Default destructor
     */
    ~InputHandler() = default;

    /**
     *  Get Current State Input
     *  @return Command Object
     */
    nextfloor::universe::commands::Command* HandlerInput() noexcept;

    /**
     *  Get HID Pointer angles changes
     *  @return HIDPointer struct
     */
    HIDPointer RecordHIDPointer() noexcept;

    /**
     *  Get FOV (FieldOfView) Current State
     *  @return FOV value
     */
    float RecordFOV() noexcept;

private:
    /** The main hid */
    std::unique_ptr<HID> hid_{nullptr};

    /** GL Window pointer */
    GLFWwindow* window_;

    /** Character Commands */
    std::unique_ptr<nextfloor::universe::commands::Command> move_up_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> move_down_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> move_left_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> move_right_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> jump_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> run_command_{nullptr};
    std::unique_ptr<nextfloor::universe::commands::Command> fire_command_{nullptr};
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_HID_INPUTHANDLER_H_
