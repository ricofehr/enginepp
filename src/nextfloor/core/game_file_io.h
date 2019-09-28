/**
 *  @file game_file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GAMEFILEIO_H_
#define NEXTFLOOR_CORE_GAMEFILEIO_H_

#include "nextfloor/core/file_io.h"

#include <fstream>
#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class GameFileIO
 *  @brief contains standard FileIO Operations for the engine
 */
class GameFileIO : public FileIO {

public:
    GameFileIO();
    ~GameFileIO() noexcept final;

    std::string ReadFile(const std::string& file_path) const final;

private:
    void HandleErrorOnReadFile(const std::string& file_path) const;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_GAMEFILEIO_H_
