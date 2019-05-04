/**
 *  @file file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILEIO_H_
#define NEXTFLOOR_CORE_FILEIO_H_

#include <fstream>
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
 *  @brief contains all FileIO Operations for the engine
 */
class FileIO {

public:

    FileIO();

    FileIO(FileIO&&) = default;

    FileIO& operator=(FileIO&&) = default;

    /**
     *  Copy constructor Deleted : Ensure a sole Instance
     */
    FileIO(const FileIO&) = delete;

    /**
     *  Copy assignement Deleted: Ensure a sole Instance
     */
    FileIO& operator=(const FileIO&) = delete;

    ~FileIO();

    std::string ReadFile(std::string file_path) const;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_FILEIO_H_