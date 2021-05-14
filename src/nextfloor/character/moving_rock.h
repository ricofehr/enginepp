/**
 *  @file moving_rock.h
 *  @brief MovingRock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CHARACTER_MOVINGROCK_H_
#define NEXTFLOOR_CHARACTER_MOVINGROCK_H_

#include "nextfloor/character/character.h"

#include <memory>

#include "nextfloor/character/fsm.h"
#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace character {

/**
 *  @class MovingRock
 *  @brief Game MovingRock implementation
 */
class MovingRock : public Character {

public:
    static constexpr float kBorderScale = 1.0f;
    static constexpr char kTexture[] = "assets/rock.jpg";

    MovingRock(std::unique_ptr<nextfloor::mesh::Border> border, std::unique_ptr<FSM> fsm,
               std::vector<std::unique_ptr<nextfloor::mesh::Polygon>> rock,
               const glm::vec3& movement);
    ~MovingRock() final = default;

    bool is_flying() const final { return true; }
    bool IsPlayer() const final { return false; }
    bool IsCamera() const final { return false; }
};

}  // namespace character

}  // namespace nextfloor

#endif  // NEXTFLOOR_CHARACTER_MOVINGROCK_H_
