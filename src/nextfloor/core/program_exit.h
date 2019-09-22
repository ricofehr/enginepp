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
    ~ProgramExit() noexcept final;

    void ExitOnError() const final;
    void ExitOnSuccess() const final;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_PROGRAMEXIT_H_
