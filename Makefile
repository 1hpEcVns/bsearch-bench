.PHONY: all bench bench-rs plot

CXX ?= g++
CXXFLAGS ?= -O3 -mavx2 -std=c++23
RUSTC ?= rustc

all: bench

bench: bench.cpp
	$(CXX) $(CXXFLAGS) bench.cpp -o bench

bench-rs: bench.rs
	$(RUSTC) -O -C target-cpu=native bench.rs -o bench_rs

bench-run: bench
	taskset -c 0 ./bench > results.csv

bench-run-rs: bench-rs
	taskset -c 0 ./bench_rs > results_rs.csv

plot: bench-run
	nix develop --command python3 plot.py
