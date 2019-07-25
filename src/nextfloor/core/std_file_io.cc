/**
 *  @file std_file_io.cc
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/std_file_io.h"

#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

StdFileIO::StdFileIO()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

std::string StdFileIO::ReadFile(std::string file_path) const
{
    std::string file_str;
    std::ifstream file_stream(file_path, std::ios::in);
    if (file_stream.is_open()) {
        std::string line = "";
        while(getline(file_stream, line)) {
            file_str += "\n" + line;
        }
        file_stream.close();
    } else {
        std::cerr << "Impossible to open " << file_path << std::endl;
        getchar();
        return "";
    }

    return file_str;
}

StdFileIO::~StdFileIO()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor
