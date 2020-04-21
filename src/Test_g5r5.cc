/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./Test_g5r5.cc

    Copyright (C) 2015 - 2020

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

#include <Grid/Grid.h>
#include <Multigrid.h>
#include <Benchmark_helpers.h>

using namespace Grid;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::Rework;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);
  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate blockSize = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({2, 2, 2, 2}));

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  const int Ls = 16;
  const int nBasis = 32;
  const int nB = nBasis / 2;

  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian*         FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::cout << GridLogMessage << "UGrid:" << std::endl; UGrid->show_decomposition();
  std::cout << GridLogMessage << "UrbGrid:" << std::endl; UrbGrid->show_decomposition();
  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "FrbGrid:" << std::endl; FrbGrid->show_decomposition();

  GridParallelRNG RNG4(UGrid);
  GridParallelRNG RNG5(FGrid);

  std::vector<int> seeds4({1, 2, 3, 4});
  std::vector<int> seeds5({1, 2, 3, 4});

  RNG4.SeedFixedIntegers(seeds4);
  RNG5.SeedFixedIntegers(seeds5);

  std::vector<LatticeFermion> basis(nBasis, FGrid);
  std::vector<LatticeFermion> basisSave(nB, FGrid);

  /////////////////////////////////////////////////////////////////////////////
  //                              Test for G5R5                              //
  /////////////////////////////////////////////////////////////////////////////

  {
    LatticeFermion tmp1(FGrid);
    LatticeFermion tmp2(FGrid);
    for(int n = 0; n < nB; n++) {
      random(RNG5, tmp1);
      basisSave[n] = tmp1;
      G5R5(tmp2, tmp1);
      axpby(basis[n], 0.5, 0.5, tmp1, tmp2);       // (1 + G5R5)/2
      axpby(basis[n + nB], 0.5, -0.5, tmp1, tmp2); // (1 - G5R5)/2
      std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
                << "norm2(vec[" << n << "]) = " << norm2(basis[n]) << ". "
                << "norm2(vec[" << n + nB << "]) = " << norm2(basis[n + nB]) << std::endl;
    }

    LatticeFermion sum(FGrid);
    for(int n = 0; n < nB; n++) {
      sum = basis[n] + basis[n + nB];
      printDeviationFromReference(1e-15, basisSave[n], sum);
    }
  }

  std::cout << GridLogMessage << "Finished tests for G5R5" << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                             Test for G5CR5                              //
  /////////////////////////////////////////////////////////////////////////////

  {
    LatticeFermion tmp1(FGrid);
    LatticeFermion tmp2(FGrid);
    for(int n = 0; n < nB; n++) {
      random(RNG5, tmp1);
      basisSave[n] = tmp1;
      G5CR5(tmp2, tmp1);
      axpby(basis[n], 0.5, 0.5, tmp1, tmp2);       // (1 + G5CR5)/2
      axpby(basis[n + nB], 0.5, -0.5, tmp1, tmp2); // (1 - G5CR5)/2
      std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
                << "norm2(vec[" << n << "]) = " << norm2(basis[n]) << ". "
                << "norm2(vec[" << n + nB << "]) = " << norm2(basis[n + nB]) << std::endl;
    }

    LatticeFermion sum(FGrid);
    for(int n = 0; n < nB; n++) {
      sum = basis[n] + basis[n + nB];
      printDeviationFromReference(1e-15, basisSave[n], sum);
    }
  }

  std::cout << GridLogMessage << "Finished tests for G5CR5" << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //             Test for G5R5 in 4d (remove, as it doesn't work)            //
  /////////////////////////////////////////////////////////////////////////////

  LatticeFermion src(UGrid); random(RNG4, src);
  LatticeFermion ref(UGrid); ref = Zero();
  LatticeFermion res(UGrid); res = Zero();

  Gamma G5(Gamma::Algebra::Gamma5);

  ref = G5 * src;
  G5R5(res, src);

  printDeviationFromReference(1e-15, ref, res);

  Grid_finalize();
}
