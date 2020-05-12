/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/multigrid/Test_wilsonclover_mg_mp.cc

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
#include <Multigrid.h>

using namespace std;
using namespace Grid;
using namespace Grid::Rework;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_wilsonclover_mg_mp
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // clang-format off
  GridCartesian         *FGrid_d   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridCartesian         *FGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *FrbGrid_d = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid_d);
  GridRedBlackCartesian *FrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid_f);
  // clang-format on

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

  LevelInfo levelInfo_d(FGrid_d, params.mg);
  LevelInfo levelInfo_f(FGrid_f, params.mg);

  const int nbasis = NBASIS;
#if !defined(USE_NEW_COARSENING)
  static_assert((nbasis & 0x1) == 0, "");
#endif

  std::cout << GridLogMessage << "Compiled with nBasis = " << nbasis << " -> nB = " << nbasis / 2 << std::endl;

  std::vector<int> fSeeds({1, 2, 3, 4});
  GridParallelRNG  fPRNG(FGrid_d);
  fPRNG.SeedFixedIntegers(fSeeds);

  // clang-format off
  LatticeFermionD       src_d(FGrid_d);
  LatticeFermionD resultMGD_d(FGrid_d); resultMGD_d = Zero();
  LatticeFermionD resultMGF_d(FGrid_d); resultMGF_d = Zero();
  LatticeGaugeFieldD    Umu_d(FGrid_d);
  LatticeGaugeFieldF    Umu_f(FGrid_f);
  // clang-format on

  if(params.test.sourceType == "ones")
    src_d = 1.;
  else if(params.test.sourceType == "random")
    random(fPRNG, src_d);
  else if(params.test.sourceType == "gaussian")
    gaussian(fPRNG, src_d);

  if(params.test.config != "foo") {
    FieldMetaData header;
    IldgReader    _IldgReader;
    _IldgReader.open(params.test.config);
    _IldgReader.readConfiguration(Umu_f, header);
    precisionChange(Umu_d, Umu_f);
    _IldgReader.close();
  } else
    SU3::HotConfiguration(fPRNG, Umu_d);
    precisionChange(Umu_f, Umu_d);

  typename WilsonCloverFermionD::ImplParams implParams_d;
  typename WilsonCloverFermionF::ImplParams implParams_f;
  WilsonAnisotropyCoefficients              anisParams;

  std::vector<Complex> boundary_phases(Nd, 1.); // default is periodic
  if(params.test.useAntiPeriodicBC)
    boundary_phases[Nd-1] = -1.;
  implParams_d.boundary_phases = boundary_phases;
  implParams_f.boundary_phases = boundary_phases;

  WilsonCloverFermionD Dwc_solve_d(Umu_d, *FGrid_d, *FrbGrid_d, params.test.massSolve, params.test.csw, params.test.csw, anisParams, implParams_d);
  WilsonCloverFermionF Dwc_solve_f(Umu_f, *FGrid_f, *FrbGrid_f, params.test.massSolve, params.test.csw, params.test.csw, anisParams, implParams_f);
  WilsonCloverFermionD Dwc_setup_d(Umu_d, *FGrid_d, *FrbGrid_d, params.test.massSetup, params.test.csw, params.test.csw, anisParams, implParams_d);
  WilsonCloverFermionF Dwc_setup_f(Umu_f, *FGrid_f, *FrbGrid_f, params.test.massSetup, params.test.csw, params.test.csw, anisParams, implParams_f);

  MdagMLinearOperator<WilsonCloverFermionD, LatticeFermionD> MdagMOpDwc_d(Dwc_solve_d);
  MdagMLinearOperator<WilsonCloverFermionF, LatticeFermionF> MdagMOpDwc_f(Dwc_solve_f);

  std::cout << GridLogMessage << "**************************************************" << std::endl;
  std::cout << GridLogMessage << "Testing single-precision Multigrid for Wilson Clover" << std::endl;
  std::cout << GridLogMessage << "**************************************************" << std::endl;

#if defined(USE_NEW_COARSENING)
  auto MGPreconDwc_f = createMGInstance<vSpinColourVectorF,  vComplexF, nbasis / 2, 2, WilsonCloverFermionF>(params.mg, levelInfo_f, Dwc_setup_f, Dwc_setup_f);
#else // corresponds to original coarsening, here nCoarseSpins is equal to 1
  auto MGPreconDwc_f = createMGInstance<vSpinColourVectorF, vTComplexF, nbasis,     1, WilsonCloverFermionF>(params.mg, levelInfo_f, Dwc_setup_f, Dwc_setup_f);
#endif

  bool doRunChecks = GridCmdOptionExists(argv, argv + argc, "--runchecks");

  MGPreconDwc_f->initialSetup();
  if(doRunChecks) MGPreconDwc_f->runChecks(1e-5);

  MGPreconDwc_f->refinementSetup();
  if(doRunChecks) MGPreconDwc_f->runChecks(1e-5);

  MixedPrecisionFlexibleGeneralisedMinimalResidual<LatticeFermionD, LatticeFermionF> MPFGMRESPREC(params.test.outerSolverTol, params.test.outerSolverMaxIter, FGrid_f, *MGPreconDwc_f, params.test.outerSolverRestartLength, false);

  GridStopWatch solveTimer;
  solveTimer.Reset();
  std::cout << std::endl << "Starting with a new solver" << std::endl;
  solveTimer.Start();
  MPFGMRESPREC(MdagMOpDwc_d, src_d, resultMGF_d);
  solveTimer.Stop();
  std::cout << GridLogMessage << "Solver took: " << solveTimer.Elapsed() << std::endl;
  solveTimer.Reset();

  MGPreconDwc_f->reportTimings();

  if(GridCmdOptionExists(argv, argv + argc, "--docomparison")) {

    std::cout << GridLogMessage << "**************************************************" << std::endl;
    std::cout << GridLogMessage << "Testing double-precision Multigrid for Wilson Clover" << std::endl;
    std::cout << GridLogMessage << "**************************************************" << std::endl;

#if defined(USE_NEW_COARSENING)
    auto MGPreconDwc_d = createMGInstance<vSpinColourVectorD,  vComplexD, nbasis / 2, 2, WilsonCloverFermionD>(params.mg, levelInfo_d, Dwc_setup_d, Dwc_setup_d);
#else // corresponds to original coarsening, here nCoarseSpins is equal to 1
    auto MGPreconDwc_d = createMGInstance<vSpinColourVectorD, vTComplexD, nbasis,     1, WilsonCloverFermionD>(params.mg, levelInfo_d, Dwc_setup_d, Dwc_setup_d);
#endif

    MGPreconDwc_d->initialSetup();
    if(doRunChecks) MGPreconDwc_d->runChecks(1e-13);

    MGPreconDwc_d->refinementSetup();
    if(doRunChecks) MGPreconDwc_d->runChecks(1e-13);

    FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRESPREC(
      params.test.outerSolverTol, params.test.outerSolverMaxIter, *MGPreconDwc_d, params.test.outerSolverRestartLength, false);

    solveTimer.Reset();
    std::cout << std::endl << "Starting with a new solver" << std::endl;
    solveTimer.Start();
    FGMRESPREC(MdagMOpDwc_d, src_d, resultMGD_d);
    solveTimer.Stop();
    std::cout << GridLogMessage << "Solver took: " << solveTimer.Elapsed() << std::endl;

    MGPreconDwc_d->reportTimings();

    std::cout << GridLogMessage << "**************************************************" << std::endl;
    std::cout << GridLogMessage << "Comparing single-precision Multigrid with double-precision one for Wilson Clover" << std::endl;
    std::cout << GridLogMessage << "**************************************************" << std::endl;

    LatticeFermionD diffFullSolver(FGrid_d);

    RealD deviationFullSolver = axpy_norm(diffFullSolver, -1.0, resultMGF_d, resultMGD_d);

    // clang-format off
    LatticeFermionF src_f(FGrid_f);    precisionChange(src_f, src_d);
    LatticeFermionF resMGF_f(FGrid_f); resMGF_f = Zero();
    LatticeFermionD resMGD_d(FGrid_d); resMGD_d = Zero();
    // clang-format on

    (*MGPreconDwc_f)(src_f, resMGF_f);
    (*MGPreconDwc_d)(src_d, resMGD_d);

    LatticeFermionD diffOnlyMG(FGrid_d);
    LatticeFermionD resMGF_d(FGrid_d);
    precisionChange(resMGF_d, resMGF_f);

    RealD deviationOnlyPrec = axpy_norm(diffOnlyMG, -1.0, resMGF_d, resMGD_d);

    // clang-format off
    std::cout << GridLogMessage << "Absolute difference between FGMRES preconditioned by double and single precicision MG: " << deviationFullSolver                      << std::endl;
    std::cout << GridLogMessage << "Relative deviation  between FGMRES preconditioned by double and single precicision MG: " << deviationFullSolver / norm2(resultMGD_d) << std::endl;
    std::cout << GridLogMessage << "Absolute difference between one iteration of MG Prec in double and single precision:   " << deviationOnlyPrec                        << std::endl;
    std::cout << GridLogMessage << "Relative deviation  between one iteration of MG Prec in double and single precision:   " << deviationOnlyPrec / norm2(resMGD_d)      << std::endl;
    // clang-format on
  }

  Grid_finalize();
}
