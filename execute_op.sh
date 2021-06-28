#!/usr/bin/env bash
cd codegen
python matmul_tik.py
cp -r kernel_meta/ ../run/out/
cd -
mkdir -p build/intermediates/host/
cd build/intermediates/host/
cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd -
cd run/out
./execute_matmul_op
cd -

