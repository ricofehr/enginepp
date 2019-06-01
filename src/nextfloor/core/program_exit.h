/**
 *  @file exit_program.h
 *  @brief Manages Program Exits
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_PROGRAMEXIT_H_
#define NEXTFLOOR_CORE_PROGRAMEXIT_H_

#include "nextfloor/core/exit.h"

namespace nextfloor {

namespace core {

/**
 *  @class ExitProgram
 *  @brief Manages Program Exits
 */
class ProgramExit : public Exit {

public:

    ProgramExit();

    ProgramExit(ProgramExit&&) = default;

    ProgramExit& operator=(ProgramExit&&) = default;

    /* Copy constructor Deleted : Ensure a sole Instance */
    ProgramExit(const ProgramExit&) = delete;

    /* Copy assignement Deleted: Ensure a sole Instance */
    ProgramExit& operator=(const ProgramExit&) = delete;

    virtual ~ProgramExit() override final;

    virtual void ExitOnError() const noexcept override final;

    virtual void ExitOnSuccess() const noexcept override final;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_PROGRAMEXIT_H_
