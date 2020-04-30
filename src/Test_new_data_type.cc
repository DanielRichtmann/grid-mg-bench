/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/Test_new_data_type.cc

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
#include <tests/multigrid/Multigrid.h>

using namespace Grid;
using namespace Grid::Rework;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_new_data_type
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);
  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  const int nBasis     = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int nB         = nBasis / 2;
  Coordinate blockSize = Coordinate({2, 2, 2, 2});
  // clang-format on

  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian*         FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian*         CGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);
  GridRedBlackCartesian* CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "CGrid:" << std::endl; CGrid->show_decomposition();
  std::cout << GridLogMessage << "FrbGrid:" << std::endl; FrbGrid->show_decomposition();
  std::cout << GridLogMessage << "CrbGrid:" << std::endl; CrbGrid->show_decomposition();

  GridParallelRNG FPRNG(FGrid);
  GridParallelRNG CPRNG(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  FPRNG.SeedFixedIntegers(seeds);
  CPRNG.SeedFixedIntegers(seeds);

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(FGrid);
  SU3::HotConfiguration(FPRNG, Umu);

  RealD mass = 0.5;

  WilsonFermionR                                      Dw(Umu, *FGrid, *FrbGrid, mass);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> LinOp(Dw);

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningPolicy<LatticeFermion, nB, 1> OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2> TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4> FourSpinCoarseningPolicy;

  typedef TwoSpinCoarseningPolicy::FineFermionFieldSplit SplitFermionField;

  /////////////////////////////////////////////////////////////////////////////
  //                       Start of actual testing code                      //
  /////////////////////////////////////////////////////////////////////////////

  SplitFermionField src(FGrid); random(FPRNG, src);
  SplitFermionField res(FGrid); res = Zero();

  LinOp.Op(src, res);

  Grid_finalize();
}
