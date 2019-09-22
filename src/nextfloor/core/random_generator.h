/**
 *  @file random_number.h
 *  @brief Abstract class who defines random generator operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_RANDOMGENERATOR_H_
#define NEXTFLOOR_CORE_RANDOMGENERATOR_H_

namespace nextfloor {

namespace core {

/**
 *  @class RandomGenerator
 *  @brief Pure interface who defines random generator operations
 */
class RandomGenerator {

public:
    virtual ~RandomGenerator() = default;

    virtual int GenerateNumber() const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_RANDOMGENERATOR_H_
