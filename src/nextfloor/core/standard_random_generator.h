/**
 *  @file standard_random_generator.h
 *  @brief Generates Random Number with standard library
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_STANDARDRANDOMGENERATOR_H_
#define NEXTFLOOR_CORE_STANDARDRANDOMGENERATOR_H_

#include "nextfloor/core/random_generator.h"

namespace nextfloor {

namespace core {

/**
 *  @class StandardRandomGenerator
 *  @brief Generates Random Number with standard library
 */
class StandardRandomGenerator : public RandomGenerator {

public:

    StandardRandomGenerator();

    StandardRandomGenerator(StandardRandomGenerator&&) = default;

    StandardRandomGenerator& operator=(StandardRandomGenerator&&) = default;

    /* Copy constructor Deleted : Ensure a sole Instance */
    StandardRandomGenerator(const StandardRandomGenerator&) = delete;

    /* Copy assignement Deleted: Ensure a sole Instance */
    StandardRandomGenerator& operator=(const StandardRandomGenerator&) = delete;

    virtual ~StandardRandomGenerator() override final;

    virtual int GenerateNumber() const override final;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_STANDARDRANDOMGENERATOR_H_
