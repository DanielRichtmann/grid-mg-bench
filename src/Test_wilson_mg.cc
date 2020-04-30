/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/multigrid/Test_wilson_mg.cc

    Copyright (C) 2015-2018

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
#include <tests/multigrid/Multigrid.h>

using namespace std;
using namespace Grid;
using namespace Grid::Rework;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_wilson_mg
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char **argv) {

  Grid_init(&argc, &argv);

  GridCartesian *        FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);

  MGTestParams params;

  if(GridCmdOptionExists(argv, argv + argc, "--inputxml")) {
    std::string inputXml = GridCmdOptionPayload(argv, argv + argc, "--inputxml");
    assert(inputXml.length() != 0);

    XmlReader reader(inputXml);
    read(reader, "Params", params);
    std::cout << GridLogMessage << "Read in " << inputXml << std::endl;
  }

  // {
  //   XmlWriter writer("mg_params_template.xml");
  //   write(writer, "Params", params);
  //   std::cout << GridLogMessage << "Written mg_params_template.xml" << std::endl;
  // }

  checkParameterValidity(params);
  std::cout << params << std::endl;

  LevelInfo levelInfo(FGrid, params.mg);

  const int nbasis = NBASIS;
#if !defined(USE_NEW_COARSENING)
  static_assert((nbasis & 0x1) == 0, "");
#endif

  std::cout << GridLogMessage << "Compiled with nBasis = " << nbasis << " -> nB = " << nbasis / 2 << std::endl;

  std::vector<int> fSeeds({1, 2, 3, 4});
  GridParallelRNG  fPRNG(FGrid);
  fPRNG.SeedFixedIntegers(fSeeds);

  LatticeFermion    src(FGrid);
  LatticeFermion    result(FGrid);
  LatticeGaugeField Umu(FGrid);

  if(params.test.sourceType == "ones")
    src = 1.;
  else if(params.test.sourceType == "random")
    random(fPRNG, src);
  else if(params.test.sourceType == "gaussian")
    gaussian(fPRNG, src);

  if(params.test.config != "foo") {
    FieldMetaData      header;
    IldgReader         _IldgReader;
    _IldgReader.open(params.test.config);
    _IldgReader.readConfiguration(Umu, header);
    _IldgReader.close();
  } else
    SU3::HotConfiguration(fPRNG, Umu);

  typename WilsonFermionR::ImplParams implParams;

  std::vector<Complex> boundary_phases(Nd, 1.); // default is periodic
  if(params.test.useAntiPeriodicBC)
    boundary_phases[Nd-1] = -1.;
  implParams.boundary_phases = boundary_phases;

  WilsonFermionR Dw(Umu, *FGrid, *FrbGrid, params.test.massSolve, implParams);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDw(Dw);

  TrivialPrecon<LatticeFermion> TrivialPrecon;

  std::cout << GridLogMessage << "**************************************************" << std::endl;
  std::cout << GridLogMessage << "Testing Multigrid for Wilson" << std::endl;
  std::cout << GridLogMessage << "**************************************************" << std::endl;

#if defined(USE_NEW_COARSENING)
  auto MGPreconDw = createMGInstance<vSpinColourVector,  vComplex, nbasis / 2, 2, WilsonFermionR>(params.mg, levelInfo, Dw, Dw);
#else // corresponds to original coarsening, here nCoarseSpins is equal to 1
  auto MGPreconDw = createMGInstance<vSpinColourVector, vTComplex, nbasis,     1, WilsonFermionR>(params.mg, levelInfo, Dw, Dw);
#endif

  bool  doRunChecks          = (GridCmdOptionExists(argv, argv + argc, "--runchecks")) ? true : false;
  RealD toleranceForMGChecks = (getPrecision<LatticeFermion>::value == 1) ? 1e-6 : 1e-13;

  MGPreconDw->initialSetup();
  if(doRunChecks) MGPreconDw->runChecks(toleranceForMGChecks);

  MGPreconDw->refinementSetup();
  if(doRunChecks) MGPreconDw->runChecks(toleranceForMGChecks);

  std::vector<std::unique_ptr<OperatorFunction<LatticeFermion>>> solversDw;

  solversDw.emplace_back(new ConjugateGradient<LatticeFermion>(params.test.outerSolverTol, 10000, false));
  solversDw.emplace_back(new FlexibleGeneralisedMinimalResidual<LatticeFermion>(params.test.outerSolverTol, 10000, TrivialPrecon, params.test.outerSolverRestartLength, false));
  solversDw.emplace_back(new FlexibleGeneralisedMinimalResidual<LatticeFermion>(params.test.outerSolverTol, params.test.outerSolverMaxIter, *MGPreconDw, params.test.outerSolverRestartLength, false));

  GridStopWatch solveTimer;
  for(auto const &solver : solversDw) {
    solveTimer.Reset();
    std::cout << std::endl << "Starting with a new solver" << std::endl;
    result = Zero();
    solveTimer.Start();
    (*solver)(MdagMOpDw, src, result);
    solveTimer.Stop();
    std::cout << GridLogMessage << "Solver took: " << solveTimer.Elapsed() << std::endl;
  }

  MGPreconDw->reportTimings();

  Grid_finalize();
}
