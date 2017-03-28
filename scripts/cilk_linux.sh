#!/bin/bash
#
# Install cilkplus library and clang specific version
# @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)

if [[ -d /usr/local/cilkplus ]]; then
    echo "Cilkplus is already installed"
    exit 0
fi

sudo mkdir /usr/local/cilkplus
sudo chmod 777 /usr/local/cilkplus

rm -rf /tmp/cilkplus
mkdir /tmp/cilkplus
pushd /tmp/cilkplus >/dev/null
git clone -b cilkplus https://github.com/cilkplus/llvm llvm
git clone -b cilkplus https://github.com/cilkplus/clang llvm/tools/clang
git clone -b cilkplus https://github.com/cilkplus/compiler-rt llvm/projects/compiler-rt

pushd llvm >/dev/null
mkdir build
pushd build >/dev/null

cmake -G "Unix Makefiles" -DINTEL_SPECIFIC_CILKPLUS=1 -DCMAKE_INSTALL_PREFIX=/usr/local/cilkplus -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 ..
make && make install
popd >/dev/null
popd >/dev/null
curl -OsSL "https://cilkplus.github.io/binaries/archive-linux.tar.gz"
tar xvfz archive-linux.tar.gz
pushd archive-linux >/dev/null
rsync -av lib/cilk /usr/local/cilkplus/include/
rsync -av lib/libcilk* /usr/local/cilkplus/lib/
popd >/dev/null
popd >/dev/null

rm -rf /tmp/cilkplus
