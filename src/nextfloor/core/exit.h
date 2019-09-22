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
 *  @brief pure interface who defines Exit operations
 */
class Exit {

public:
    virtual ~Exit() = default;

    virtual void ExitOnError() const = 0;
    virtual void ExitOnSuccess() const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_EXITPROGRAM_H_
