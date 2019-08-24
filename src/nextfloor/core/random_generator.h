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
 *  @brief Abstract class who defines random generator operations
 */
class RandomGenerator {

public:
    RandomGenerator(RandomGenerator&&) = default;
    RandomGenerator& operator=(RandomGenerator&&) = default;
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;

    virtual ~RandomGenerator() = default;

    virtual int GenerateNumber() const = 0;

protected:
    RandomGenerator() = default;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_RANDOMGENERATOR_H_
