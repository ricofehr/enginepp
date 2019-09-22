/**
 *  @file file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILEIO_H_
#define NEXTFLOOR_CORE_FILEIO_H_

#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class FileIO
 *  @brief Abstract class for file operations
 */
class FileIO {

public:
    virtual ~FileIO() = default;

    virtual std::string ReadFile(const std::string& file_path) const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_FILEIO_H_
