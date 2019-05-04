/**
 *  @file random_number.h
 *  @brief Generates Random Number
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_RANDOMNUMBER_H_
#define NEXTFLOOR_CORE_RANDOMNUMBER_H_

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
 *  @class RandomNumber
 *  @brief Generates Random Number
 */
class RandomNumber {

public:

    RandomNumber();

    RandomNumber(RandomNumber&&) = default;

    RandomNumber& operator=(RandomNumber&&) = default;

    /**
     *  Copy constructor Deleted : Ensure a sole Instance
     */
    RandomNumber(const RandomNumber&) = delete;

    /**
     *  Copy assignement Deleted: Ensure a sole Instance
     */
    RandomNumber& operator=(const RandomNumber&) = delete;

    ~RandomNumber();
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_RANDOMNUMBER_H_