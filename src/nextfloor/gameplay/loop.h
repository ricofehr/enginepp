/**
 *  @file loop.h
 *  @brief Loop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_LOOP_H_
#define NEXTFLOOR_GAMEPLAY_LOOP_H_

namespace nextfloor {

namespace gameplay {

/**
 *  @class Loop
 *  @brief  Loop manages the lifetime of the opengl scene
 */
class Loop {

public:
    virtual ~Loop() = default;

    virtual void RunLoop() = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_LOOP_H_
