/**
 *  @file game_menu_factory.h
 *  @brief Cincrete Factory Class Implementation for game menus
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MENU_GAMEMENUFACTORY_H_
#define NEXTFLOOR_MENU_GAMEMENUFACTORY_H_

#include "nextfloor/gameplay/menu_factory.h"

namespace nextfloor {

namespace menu {

/**
 *  @class GameMenuFactory
 *  @brief Concrete Factory for game menus
 */
class GameMenuFactory : public nextfloor::gameplay::MenuFactory {

public:
    GameMenuFactory() = default;
    ~GameMenuFactory() final = default;

    std::unique_ptr<nextfloor::gameplay::Menu> MakeMainMenu() const final;
};

}  // namespace menu

}  // namespace nextfloor

#endif  // NEXTFLOOR_MENU_GAMEMENUFACTORY_H_
