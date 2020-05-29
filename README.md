# grid-mg-bench

A repository I use to develop, test, and benchmark multigrid-related things based on [Grid](https://github.com/paboyle/Grid).

## Building/Installation

This repo uses Grid's `grid-config` binary. Installation is done via:

```shell
mkdir <build_dir> && cd <build_dir>
cmake -DGRID_DIR=/path/to/grid/install/dir
make
make install
```

Note: The current `cmake` setup doesn't work together nicely with how Grid uses `nvcc`. Hence when compiling for GPUs, I use the following after the call to `cmake` as a temporary workaround:

```shell
cd <build_dir>
find . -name link.txt -exec sed -i 's|"nvcc -x cu"|nvcc -link|g' {} \;
grep -rl '"nvcc -x cu"' | xargs -I{} sed -i 's|"nvcc -x cu"|nvcc -x cu|g' {}
```
