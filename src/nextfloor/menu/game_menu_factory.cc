/**
 *  @file game_menu_factory.c
 *  @brief Concrete Factory Class for Game Menus
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/menu/game_menu_factory.h"

#include <memory>

#include "nextfloor/menu/main_menu.h"

namespace nextfloor {

namespace menu {

std::unique_ptr<nextfloor::gameplay::Menu> GameMenuFactory::MakeMainMenu() const
{
    return std::make_unique<MainMenu>();
}

}  // namespace menu

}  // namespace nextfloor
