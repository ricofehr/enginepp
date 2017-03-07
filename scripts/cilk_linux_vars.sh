# clang binaries
export PATH=/usr/local/cilkplus/bin:$PATH
# cilkplus headers
export C_INCLUDE_PATH=/usr/local/cilkplus/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cilkplus/include:$CPLUS_INCLUDE_PATH
# libraires path
export LIBRARY_PATH=/usr/local/cilkplus/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cilkplus/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/cilkplus/lib:$DYLD_LIBRARY_PATH

# C compiler
export CC=/usr/local/cilkplus/bin/clang

# C++ compiler
export CXX=/usr/local/cilkplus/bin/clang++
