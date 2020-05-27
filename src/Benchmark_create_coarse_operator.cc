/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_create_coarse_operator.cc

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
#include <CoarsenedMatrixBaseline.h>
#include <CoarsenedMatrixUpstream.h>
#include <CoarsenedMatrixUpstreamImprovedDirsave.h>
#include <CoarsenedMatrixUpstreamImprovedDirsaveLut.h>
#include <CoarsenedMatrixUpstreamImprovedDirsaveLutMRHS.h>
#include <Benchmark_helpers.h>
#include <Layout_converters.h>

using namespace Grid;
using namespace Grid::Rework;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::LayoutConverters;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_coarsenedmatrix
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  const int  nBasis              = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int  nB                  = nBasis / 2;
        int  nParSetupVecs       = readFromCommandLineInt(&argc, &argv, "--parsetupvecs", nB);
  Coordinate blockSize           = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({4, 4, 4, 4}));
  bool       runAll              = readFromCommandLineToggle(&argc, &argv, "--all");
  std::vector<std::string> toRun = readFromCommandLineCSL(&argc, &argv, "--torun", {"Speed2FastProj"});
  // clang-format on

  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                            Print warning/info                           //
  /////////////////////////////////////////////////////////////////////////////

  if(!GridCmdOptionExists(argv, argv + argc, "--torun") &&
     !GridCmdOptionExists(argv, argv + argc, "--all")) {
    std::cout << GridLogWarning << "You did not specify argument --torun. Only benchmark for Upstream CoarsenOperator will be performed" << std::endl;
    std::cout << GridLogWarning << "To run more use --torun <list> with <list> being a comma separated list from" << std::endl;
    std::cout << GridLogWarning << "Baseline,ImprovedDirsave,ImprovedDirsaveLut,ImprovedDirsaveLutMRHS,Speed0SlowProj,Speed0FastProj,Speed1SlowProj,Speed1FastProj,Speed2SlowProj,Speed2FastProj" << std::endl;
    std::cout << GridLogWarning << "You can also use --all to benchmark CoarsenOperator for all implementations." << std::endl;
  }

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

#if defined(FIVE_DIMENSIONS) // 5d use case (u = gauge, f = fermion = fine, t = tmp, c = coarse)
  const int              Ls      = 16;
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian*         FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);
  GridCartesian*         TGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* TrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(TGrid);
  GridCartesian*         CGrid   = SpaceTimeGrid::makeFiveDimGrid(1, TGrid);
  GridRedBlackCartesian* CrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(1, TGrid);
#else // 4d use case (f = fine, c = coarse), fermion same as gauge
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian*         FGrid   = UGrid;
  GridRedBlackCartesian* FrbGrid = UrbGrid;
  GridCartesian*         CGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);
#endif

  UGrid->show_decomposition();
  FGrid->show_decomposition();
  CGrid->show_decomposition();

  GridParallelRNG UPRNG(UGrid);
  GridParallelRNG FPRNG(FGrid);
  GridParallelRNG CPRNG(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  UPRNG.SeedFixedIntegers(seeds);
  FPRNG.SeedFixedIntegers(seeds);
  CPRNG.SeedFixedIntegers(seeds);

  RealD tol = getPrecision<vComplex>::value == 2 ? 1e-15 : 1e-7;

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

#if defined(FIVE_DIMENSIONS)
  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(UPRNG, Umu);

  RealD mass = 0.001;
  RealD M5   = 1.8;

  DomainWallFermionR Ddwf(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5);
  MdagMLinearOperator<DomainWallFermionR,LatticeFermion> LinOp(Ddwf);
#else
  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG, Umu);

  RealD mass = 0.5;
  RealD csw = 1.0;

  WilsonFermionR       Dw(Umu, *FGrid, *FrbGrid, mass);
  WilsonCloverFermionR Dwc(Umu, *FGrid, *FrbGrid, mass, csw, csw);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion> LinOp(Dwc);

  GridCartesian*         FGrid5d   = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, FGrid);
  GridRedBlackCartesian* FrbGrid5d = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, FGrid);
  GridCartesian*         CGrid5d   = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, CGrid);
  GridRedBlackCartesian* CrbGrid5d = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, CGrid);

  WilsonMRHSFermionR       Dw5(Umu, *FGrid5d, *FrbGrid5d, *FGrid, *FrbGrid, mass);
  WilsonCloverMRHSFermionR Dwc5(Umu, *FGrid5d, *FrbGrid5d, *FGrid, *FrbGrid, mass, csw, csw);

  MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion> LinOp5(Dwc5);
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningPolicy<LatticeFermion, nB, 1> OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2> TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4> FourSpinCoarseningPolicy;

  typedef Grid::Upstream::Aggregation<vSpinColourVector, vTComplex, nBasis>                       UpstreamAggregation;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis>                       BaselineAggregation;
  typedef Grid::UpstreamImprovedDirsave::Aggregation<vSpinColourVector, vTComplex, nBasis>        ImprovedDirsaveAggregation;
  typedef Grid::UpstreamImprovedDirsaveLut::Aggregation<vSpinColourVector, vTComplex, nBasis>     ImprovedDirsaveLutAggregation;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::Aggregation<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutMRHSAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                                      OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                                      TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>                                     FourSpinAggregation;

  typedef Grid::Upstream::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                       UpstreamCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                       BaselineCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsave::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>        ImprovedDirsaveCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLut::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>     ImprovedDirsaveLutCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutMRHSCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>                                      OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>                                      TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>                                     FourSpinCoarsenedMatrix;

  typedef UpstreamCoarsenedMatrix::CoarseVector               UpstreamCoarseVector;
  typedef BaselineCoarsenedMatrix::CoarseVector               BaselineCoarseVector;
  typedef ImprovedDirsaveCoarsenedMatrix::CoarseVector        ImprovedDirsaveCoarseVector;
  typedef ImprovedDirsaveLutCoarsenedMatrix::CoarseVector     ImprovedDirsaveLutCoarseVector;
  typedef ImprovedDirsaveLutMRHSCoarsenedMatrix::CoarseVector ImprovedDirsaveLutMRHSCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField                OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField                TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField               FourSpinCoarseVector;

  typedef UpstreamCoarsenedMatrix::CoarseMatrix               UpstreamCoarseLinkField;
  typedef BaselineCoarsenedMatrix::CoarseMatrix               BaselineCoarseLinkField;
  typedef ImprovedDirsaveCoarsenedMatrix::CoarseMatrix        ImprovedDirsaveCoarseLinkField;
  typedef ImprovedDirsaveLutCoarsenedMatrix::CoarseMatrix     ImprovedDirsaveLutCoarseLinkField;
  typedef ImprovedDirsaveLutMRHSCoarsenedMatrix::CoarseMatrix ImprovedDirsaveLutMRHSCoarseLinkField;
  typedef OneSpinCoarsenedMatrix::LinkField                   OneSpinCoarseLinkField;
  typedef TwoSpinCoarsenedMatrix::LinkField                   TwoSpinCoarseLinkField;
  typedef FourSpinCoarsenedMatrix::LinkField                  FourSpinCoarseLinkField;

  /////////////////////////////////////////////////////////////////////////////
  //                       Set values for some toggles                       //
  /////////////////////////////////////////////////////////////////////////////

  const int cb          = 0; // cb to use in aggregation
  const int checkOrthog = 1; // whether to check orthog in setup of aggregation
  const int gsPasses    = 1; // number of GS in setup of aggregation
  const int isHermitian = 0; // whether we do Petrov-Galerkin (hermitian) or Galerkin (G5-hermitian) coarsening

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamAggregation UpstreamAggs(CGrid, FGrid, cb);

  UpstreamAggs.CreateSubspaceRandom(FPRNG);
  performChiralDoubling(UpstreamAggs.subspace);
  UpstreamAggs.Orthogonalise(checkOrthog, gsPasses);

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarsenedMatrix UpstreamCMat(*CGrid, isHermitian);

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil   = UpstreamCMat.geom.npoint;
  double nAccum     = nStencil - 1;
  double FSiteElems = getSiteElems<LatticeFermion>();
  double CSiteElems = getSiteElems<UpstreamCoarseVector>();

  std::cout << GridLogDebug << "FSiteElems = " << FSiteElems << std::endl;
  std::cout << GridLogDebug << "CSiteElems = " << CSiteElems << std::endl;

  double FVolume = std::accumulate(FGrid->_fdimensions.begin(), FGrid->_fdimensions.end(), 1, std::multiplies<double>());
  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

  /////////////////////////////////////////////////////////////////////////////
  //                     Determine which runs to perform                     //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  std::vector<std::string> allRuns = {
    "Baseline",
    "ImprovedDirsave",
    "ImprovedDirsaveLut",
    "ImprovedDirsaveLutMRHS",
    "Speed0SlowProj",
    "Speed0FastProj",
    "Speed1SlowProj",
    "Speed1FastProj",
    "Speed2SlowProj",
    "Speed2FastProj"
  };
  // clang-format on

  if(runAll)
    toRun = allRuns;

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for CoarsenOperator" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Will be running benchmarks for configurations " << toRun << std::endl;

    uint64_t nIterOnce = 1;
    uint64_t nSecOnce  = 100;

    UpstreamCoarseLinkField CoarseLFUpstreamTmp(CGrid);

    double flop = 0; // TODO
    double byte = 0; // TODO

    // Upstream = Reference (always run) //////////////////////////////////////

    BenchmarkFunction(UpstreamCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, UpstreamAggs);
    auto profResults = UpstreamCMat.GetProfile(); UpstreamCMat.ResetProfile();
    prettyPrintProfiling("Upstream", profResults, profResults["CoarsenOperator.Total"].t, false);

    for(auto const& elem : toRun) {
      // Baseline = state when I started working ////////////////////////////////

      std::cout << "Running benchmark for configuration " << elem << std::endl;

      if(elem == "Baseline") {
        BaselineAggregation BaselineAggs(CGrid, FGrid, cb);
        BaselineCoarsenedMatrix BaselineCMat(*CGrid, *CrbGrid, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(BaselineCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, BaselineAggs);
        profResults = BaselineCMat.GetProfile(); BaselineCMat.ResetProfile();
        prettyPrintProfiling("Baseline", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of Baseline from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], BaselineCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving /////////////////////////////

      else if(elem == "ImprovedDirsave") {
        ImprovedDirsaveAggregation ImprovedDirsaveAggs(CGrid, FGrid, cb);
        ImprovedDirsaveCoarsenedMatrix ImprovedDirsaveCMat(*CGrid, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, ImprovedDirsaveAggs);
        profResults = ImprovedDirsaveCMat.GetProfile(); ImprovedDirsaveCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsave", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsave from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving + lut ///////////////////////

      else if(elem == "ImprovedDirsaveLut") {
        ImprovedDirsaveLutAggregation ImprovedDirsaveLutAggs(CGrid, FGrid, cb);
        ImprovedDirsaveLutCoarsenedMatrix ImprovedDirsaveLutCMat(*CGrid, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveLutAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveLutCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, ImprovedDirsaveLutAggs);
        profResults = ImprovedDirsaveLutCMat.GetProfile(); ImprovedDirsaveLutCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsaveLut", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLut from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveLutCMat.A[p]);
        }
      }

      // My improvements to upstream with MRHS //////////////////////////////////

      else if(elem == "ImprovedDirsaveLutMRHS") {
        ImprovedDirsaveLutMRHSAggregation ImprovedDirsaveLutMRHSAggs(CGrid, FGrid, cb);
        ImprovedDirsaveLutMRHSCoarsenedMatrix ImprovedDirsaveLutMRHSCMat(*FGrid5d, *FGrid, *CGrid5d, *CGrid, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveLutMRHSAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveLutMRHSCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid5d, LinOp5, ImprovedDirsaveLutMRHSAggs);
        profResults = ImprovedDirsaveLutMRHSCMat.GetProfile(); ImprovedDirsaveLutMRHSCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsaveLutMRHS", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLutMRHS from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveLutMRHSCMat.A[p]);
        }
      }

      // Twospin layout speedlevel 0, slow projects /////////////////////////////

      else if(elem == "Speed0SlowProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 0, isHermitian); // speedLevel = 0
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed0.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 0, fast projects /////////////////////////////

      else if(elem == "Speed0FastProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 0, isHermitian); // speedLevel = 0
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed0.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.FastProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 1, slow projects /////////////////////////////

      else if(elem == "Speed1SlowProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 1, isHermitian); // speedLevel = 1
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed1.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 1, fast projects /////////////////////////////

      else if(elem == "Speed1FastProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 1, isHermitian); // speedLevel = 1
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed1.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.FastProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 2, slow projects /////////////////////////////

      else if(elem == "Speed2SlowProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 2, isHermitian); // speedLevel = 2
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed2.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed2.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 2, fast projects /////////////////////////////

      else if(elem == "Speed2FastProj") {
        TwoSpinAggregation TwoSpinAggs(CGrid, FGrid, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*CGrid, *CrbGrid, 2, isHermitian); // speedLevel = 2
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid, LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed2.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed2.FastProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }
    }
  }

  Grid_finalize();
}
