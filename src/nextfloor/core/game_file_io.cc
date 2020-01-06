/**
 *  @file game_file_io.cc
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/game_file_io.h"

#include <cassert>
#include <fstream>
#include <sstream>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GameFileIO::GameFileIO()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

std::string GameFileIO::ReadFile(const std::string& file_path) const
{
    std::string file_str;
    std::ifstream file_stream(file_path, std::ios::in);
    if (file_stream.is_open()) {
        std::string line = "";
        while (getline(file_stream, line)) {
            file_str += "\n" + line;
        }
        file_stream.close();
    }
    else {
        HandleErrorOnReadFile(file_path);
    }

    return file_str;
}

void GameFileIO::HandleErrorOnReadFile(const std::string& file_path) const
{
    std::ostringstream message;
    message << "Unable to open " << file_path;
    CommonServices::getLog()->WriteLine(std::move(message));
    CommonServices::getExit()->ExitOnError();
}

GameFileIO::~GameFileIO() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace core

}  // namespace nextfloor
