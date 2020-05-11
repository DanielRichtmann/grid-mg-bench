/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_clover_term.cc

    Copyright (C) 2015 - 2019

    Author: Daniel Richtmann <daniel.richtmann@ur.de>

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

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // clang-format off
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  // clang-format on

  UGrid->show_decomposition();

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG(UGrid);
  pRNG.SeedFixedIntegers(seeds);

  // clang-format off
  LatticeFermion    src(UGrid); src = Zero();
  LatticeFermion    res(UGrid); random(pRNG, res);
  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(pRNG, Umu);
  // clang-format on

  typename WilsonFermionR::ImplParams implParams;
  WilsonAnisotropyCoefficients anisParams;


  std::vector<Complex> boundary_phases(Nd, 1.);
  if(GridCmdOptionExists(argv, argv + argc, "--antiperiodic")) boundary_phases[Nd - 1] = -1.;
  implParams.boundary_phases = boundary_phases;

  RealD mass = 0.5;
  RealD csw = 1.0;

  WilsonFermionR Dw(Umu, *UGrid, *UrbGrid, mass, implParams, anisParams);
  WilsonCloverFermionR Dwc(Umu, *UGrid, *UrbGrid, mass, csw, csw, anisParams, implParams);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDw(Dw);
  MdagMLinearOperator<WilsonCloverFermionR, LatticeFermion> MdagMOpDwc(Dwc);

  int nIter = 100;

  double tWilsonM     = 0.;
  double tWilsonDhop  = 0.;
  double tWilsonMooee = 0.;
  double tCloverM     = 0.;
  double tCloverDhop  = 0.;
  double tCloverMooee = 0.;

  tWilsonM -= usecond();
  for(int i = 0; i < nIter; ++i) Dw.M(src, res);
  tWilsonM += usecond();
  res = Zero();

  tWilsonDhop -= usecond();
  for(int i = 0; i < nIter; ++i) Dw.Dhop(src, res, DaggerNo);
  tWilsonDhop += usecond();
  res = Zero();

  tWilsonMooee -= usecond();
  for(int i = 0; i < nIter; ++i) Dw.Mooee(src, res);
  tWilsonMooee += usecond();
  res = Zero();

  tCloverM -= usecond();
  for(int i = 0; i < nIter; ++i) Dwc.M(src, res);
  tCloverM += usecond();
  res = Zero();

  tCloverDhop -= usecond();
  for(int i = 0; i < nIter; ++i) Dwc.Dhop(src, res, DaggerNo);
  tCloverDhop += usecond();
  res = Zero();

  tCloverMooee -= usecond();
  for(int i = 0; i < nIter; ++i) Dwc.Mooee(src, res);
  tCloverMooee += usecond();
  res = Zero();

  double ref = tWilsonM;

  std::cout << GridLogMessage << "Running " << nIter << " iterations of the kernels" << std::endl;

  // clang-format off
  std::cout << GridLogMessage << "Dw.M:      " << tWilsonM     << " μs = " << tWilsonM/tWilsonM     << std::endl;
  std::cout << GridLogMessage << "Dw.Dhop:   " << tWilsonDhop  << " μs = " << tWilsonDhop/tWilsonM  << " x Dw.M" << std::endl;
  std::cout << GridLogMessage << "Dw.Mooee:  " << tWilsonMooee << " μs = " << tWilsonMooee/tWilsonM << " x Dw.M" << std::endl;
  std::cout << GridLogMessage << "Dwc.M:     " << tCloverM     << " μs = " << tCloverM/tWilsonM     << " x Dw.M" << std::endl;
  std::cout << GridLogMessage << "Dwc.Dhop:  " << tCloverDhop  << " μs = " << tCloverDhop/tWilsonM  << " x Dw.M" << " = " << tCloverDhop/tCloverM  << " x Dwc.M" << std::endl;
  std::cout << GridLogMessage << "Dwc.Mooee: " << tCloverMooee << " μs = " << tCloverMooee/tWilsonM << " x Dw.M" << " = " << tCloverMooee/tCloverM << " x Dwc.M" << " = " << tCloverMooee/tWilsonMooee << " x Dw.Mooee" << std::endl;
  // clang-format on

  Grid_finalize();
}
