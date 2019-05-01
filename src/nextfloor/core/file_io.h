/**
 *  @file file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILEIO_H_
#define NEXTFLOOR_CORE_FILEIO_H_

#include <fstream>
#include <iostream>

namespace nextfloor {

namespace core {

/**
 * File I/O Operations
 */
class FileIO {

public:
    /**
     *  Constructor
     */
    FileIO() = default;

    /**
     *  Default Move constructor
     */
    FileIO(FileIO&&) = default;

    /**
     *  Default Move assignment
     */
    FileIO& operator=(FileIO&&) = default;

    /**
     *  Default Copy constructor
     */
    FileIO(const FileIO&) = default;

    /**
     *  Default Copy assignment
     */
    FileIO& operator=(const FileIO&) = default;

    /**
     *  Default destructor
     */
    ~FileIO() = default;

    /**
     * Read a file and return content
     * @param  file_path is the file to read
     * @return           content as string
     */
    std::string ReadFile(std::string file_path);
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_FILEIO_H_