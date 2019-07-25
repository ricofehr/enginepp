/**
 *  @file pseudo_random_generator.h
 *  @brief Generates Random Number with standard library
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_PSEUDORANDOMGENERATOR_H_
#define NEXTFLOOR_CORE_PSEUDORANDOMGENERATOR_H_

#include "nextfloor/core/random_generator.h"

namespace nextfloor {

namespace core {

/**
 *  @class PseudoRandomGenerator
 *  @brief Generates Random Number with standard library
 */
class PseudoRandomGenerator : public RandomGenerator {

public:

    PseudoRandomGenerator();

    PseudoRandomGenerator(PseudoRandomGenerator&&) = default;

    PseudoRandomGenerator& operator=(PseudoRandomGenerator&&) = default;

    /* Copy constructor Deleted : Ensure a sole Instance */
    PseudoRandomGenerator(const PseudoRandomGenerator&) = delete;

    /* Copy assignement Deleted: Ensure a sole Instance */
    PseudoRandomGenerator& operator=(const PseudoRandomGenerator&) = delete;

    virtual ~PseudoRandomGenerator() override final;

    virtual int GenerateNumber() const override final;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_PSEUDORANDOMGENERATOR_H_
