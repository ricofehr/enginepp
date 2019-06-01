/**
 *  @file file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILEIO_H_
#define NEXTFLOOR_CORE_FILEIO_H_

#include <iostream>

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
 *  @class FileIO
 *  @brief Abstract class for file operations
 */
class FileIO {

public:

    FileIO(FileIO&&) = default;

    FileIO& operator=(FileIO&&) = default;

    /*  Copy constructor Deleted : Ensure a sole Instance */
    FileIO(const FileIO&) = delete;

    /*  Copy assignement Deleted: Ensure a sole Instance */
    FileIO& operator=(const FileIO&) = delete;

    virtual ~FileIO() = default;

    virtual std::string ReadFile(std::string file_path) const = 0;

protected:

    FileIO() = default;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_FILEIO_H_
