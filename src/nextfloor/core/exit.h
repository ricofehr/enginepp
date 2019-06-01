/**
 *  @file exit.h
 *  @brief Abstract class who defines Exit operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_EXITPROGRAM_H_
#define NEXTFLOOR_CORE_EXITPROGRAM_H_

namespace nextfloor {

namespace core {

/**
 *  @class Exit
 *  @brief Abstract class who defines Exit operations
 */
class Exit {

public:

    Exit(Exit&&) = default;

    Exit& operator=(Exit&&) = default;

    /* Copy constructor Deleted : Ensure a sole Instance */
    Exit(const Exit&) = delete;

    /* Copy assignement Deleted: Ensure a sole Instance */
    Exit& operator=(const Exit&) = delete;

    virtual ~Exit() = default;

    virtual void ExitOnError() const noexcept = 0;

    virtual void ExitOnSuccess() const noexcept = 0;

protected:

    Exit() = default;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_EXITPROGRAM_H_
