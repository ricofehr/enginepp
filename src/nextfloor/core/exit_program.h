/**
 *  @file exit_program.h
 *  @brief Manages Program Exits
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_EXITPROGRAM_H_
#define NEXTFLOOR_CORE_EXITPROGRAM_H_

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::core
 *  @brief Core system
 */
namespace core {

/**
 *  @class ExitProgram
 *  @brief Manages Program Exits
 */
class ExitProgram {

public:

    ExitProgram();

    ExitProgram(ExitProgram&&) = default;

    ExitProgram& operator=(ExitProgram&&) = default;

    /**
     *  Copy constructor Deleted : Ensure a sole Instance
     */
    ExitProgram(const ExitProgram&) = delete;

    /**
     *  Copy assignement Deleted: Ensure a sole Instance
     */
    ExitProgram& operator=(const ExitProgram&) = delete;

    ~ExitProgram();
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_EXITPROGRAM_H_