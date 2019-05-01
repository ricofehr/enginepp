/**
 *  @file file_io.cc
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/file_io.h"

namespace nextfloor {

namespace core {

std::string FileIO::ReadFile(std::string file_path)
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

} // namespace core

} // namespace nextfloor