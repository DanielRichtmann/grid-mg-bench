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
  bool       useClover           = readFromCommandLineToggle(&argc, &argv, "--clover");
  std::vector<std::string> toRun = readFromCommandLineCSL(&argc, &argv, "--torun", {"Speed2FastProj"});
  // clang-format on

  std::cout << GridLogMessage << "Using " << (useClover ? "clover" : "wilson") << " fermions" << std::endl;
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

  GridCartesian*         UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian*         UGrid_c   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_c = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_c);
  GridCartesian*         FGrid_f   = UGrid_f;
  GridRedBlackCartesian* FrbGrid_f = UrbGrid_f;
  GridCartesian*         FGrid_c   = UGrid_c;
  GridRedBlackCartesian* FrbGrid_c = UrbGrid_c;

  UGrid_f->show_decomposition();
  FGrid_f->show_decomposition();
  UGrid_c->show_decomposition();
  FGrid_c->show_decomposition();

  GridParallelRNG UPRNG_f(UGrid_f);
  GridParallelRNG FPRNG_f(FGrid_f);
  GridParallelRNG UPRNG_c(UGrid_c);
  GridParallelRNG FPRNG_c(FGrid_c);

  std::vector<int> seeds({1, 2, 3, 4});

  UPRNG_f.SeedFixedIntegers(seeds);
  FPRNG_f.SeedFixedIntegers(seeds);
  UPRNG_c.SeedFixedIntegers(seeds);
  FPRNG_c.SeedFixedIntegers(seeds);

  RealD tol = getPrecision<vComplex>::value == 2 ? 1e-15 : 1e-7;

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
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(UGrid_f); SU3::HotConfiguration(UPRNG_f, Umu);

  RealD mass = 0.5;
  RealD csw = 1.0;

  WilsonFermionR       Dw(Umu, *FGrid_f, *FrbGrid_f, mass);
  WilsonCloverFermionR Dwc(Umu, *FGrid_f, *FrbGrid_f, mass, csw, csw);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion>  LinOpDw(Dw);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion>  LinOpDwc(Dwc);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion>* LinOp;

  GridCartesian*         FGrid_f_5d   = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, FGrid_f);
  GridRedBlackCartesian* FrbGrid_f_5d = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, FGrid_f);
  GridCartesian*         FGrid_c_5d   = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, FGrid_c);
  GridRedBlackCartesian* FrbGrid_c_5d = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, FGrid_c);

  WilsonMRHSFermionR       Dw5(Umu, *FGrid_f_5d, *FrbGrid_f_5d, *FGrid_f, *FrbGrid_f, mass);
  WilsonCloverMRHSFermionR Dwc5(Umu, *FGrid_f_5d, *FrbGrid_f_5d, *FGrid_f, *FrbGrid_f, mass, csw, csw);

  MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion> LinOpDw5(Dw5);
  MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion> LinOpDwc5(Dwc5);

  MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion>* LinOp5;

  if(useClover) {
    LinOp = &LinOpDw;
    LinOp5 = &LinOpDw5;
  } else {
    LinOp  = &LinOpDwc;
    LinOp5 = &LinOpDwc5;
  }

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

  UpstreamAggregation UpstreamAggs(FGrid_c, FGrid_f, cb);

  UpstreamAggs.CreateSubspaceRandom(FPRNG_f);
  performChiralDoubling(UpstreamAggs.subspace);
  UpstreamAggs.Orthogonalise(checkOrthog, gsPasses);

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarsenedMatrix UpstreamCMat(*FGrid_c, isHermitian);

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil   = UpstreamCMat.geom.npoint;
  double nAccum     = nStencil - 1;
  double siteElems_f = getSiteElems<LatticeFermion>();
  double siteElems_c = getSiteElems<UpstreamCoarseVector>();

  std::cout << GridLogDebug << "siteElems_f = " << siteElems_f << std::endl;
  std::cout << GridLogDebug << "siteElems_c = " << siteElems_c << std::endl;

  double FVolume_f = std::accumulate(FGrid_f->_fdimensions.begin(), FGrid_f->_fdimensions.end(), 1, std::multiplies<double>());
  double FVolume_c = std::accumulate(FGrid_c->_fdimensions.begin(), FGrid_c->_fdimensions.end(), 1, std::multiplies<double>());

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

    UpstreamCoarseLinkField CoarseLFUpstreamTmp(FGrid_c);

    double flop = 0; // TODO
    double byte = 0; // TODO

    // Upstream = Reference (always run) //////////////////////////////////////

    BenchmarkFunction(UpstreamCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, UpstreamAggs);
    auto profResults = UpstreamCMat.GetProfile(); UpstreamCMat.ResetProfile();
    prettyPrintProfiling("Upstream", profResults, profResults["CoarsenOperator.Total"].t, false);

    for(auto const& elem : toRun) {
      // Baseline = state when I started working ////////////////////////////////

      std::cout << "Running benchmark for configuration " << elem << std::endl;

      if(elem == "Baseline") {
        BaselineAggregation BaselineAggs(FGrid_c, FGrid_f, cb);
        BaselineCoarsenedMatrix BaselineCMat(*FGrid_c, *FrbGrid_c, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(BaselineCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, BaselineAggs);
        profResults = BaselineCMat.GetProfile(); BaselineCMat.ResetProfile();
        prettyPrintProfiling("Baseline", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of Baseline from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], BaselineCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving /////////////////////////////

      else if(elem == "ImprovedDirsave") {
        ImprovedDirsaveAggregation ImprovedDirsaveAggs(FGrid_c, FGrid_f, cb);
        ImprovedDirsaveCoarsenedMatrix ImprovedDirsaveCMat(*FGrid_c, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, ImprovedDirsaveAggs);
        profResults = ImprovedDirsaveCMat.GetProfile(); ImprovedDirsaveCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsave", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsave from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving + lut ///////////////////////

      else if(elem == "ImprovedDirsaveLut") {
        ImprovedDirsaveLutAggregation ImprovedDirsaveLutAggs(FGrid_c, FGrid_f, cb);
        ImprovedDirsaveLutCoarsenedMatrix ImprovedDirsaveLutCMat(*FGrid_c, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveLutAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveLutCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, ImprovedDirsaveLutAggs);
        profResults = ImprovedDirsaveLutCMat.GetProfile(); ImprovedDirsaveLutCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsaveLut", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLut from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveLutCMat.A[p]);
        }
      }

      // My improvements to upstream with MRHS //////////////////////////////////

      else if(elem == "ImprovedDirsaveLutMRHS") {
        ImprovedDirsaveLutMRHSAggregation ImprovedDirsaveLutMRHSAggs(FGrid_c, FGrid_f, cb);
        ImprovedDirsaveLutMRHSCoarsenedMatrix ImprovedDirsaveLutMRHSCMat(*FGrid_f_5d, *FGrid_f, *FGrid_c_5d, *FGrid_c, isHermitian);
        for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveLutMRHSAggs.subspace[i] = UpstreamAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveLutMRHSCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f_5d, *LinOp5, ImprovedDirsaveLutMRHSAggs);
        profResults = ImprovedDirsaveLutMRHSCMat.GetProfile(); ImprovedDirsaveLutMRHSCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsaveLutMRHS", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLutMRHS from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCMat.A[p], ImprovedDirsaveLutMRHSCMat.A[p]);
        }
      }

      // Twospin layout speedlevel 0, slow projects /////////////////////////////

      else if(elem == "Speed0SlowProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 0, isHermitian); // speedLevel = 0
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed0.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 0, fast projects /////////////////////////////

      else if(elem == "Speed0FastProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 0, isHermitian); // speedLevel = 0
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed0.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.FastProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 1, slow projects /////////////////////////////

      else if(elem == "Speed1SlowProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 1, isHermitian); // speedLevel = 1
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed1.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 1, fast projects /////////////////////////////

      else if(elem == "Speed1FastProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 1, isHermitian); // speedLevel = 1
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed1.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.FastProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 2, slow projects /////////////////////////////

      else if(elem == "Speed2SlowProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 0); // 0 = don't use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 2, isHermitian); // speedLevel = 2
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
        profResults = TwoSpinCMat.GetProfile(); TwoSpinCMat.ResetProfile();
        prettyPrintProfiling("TwoSpin.Speed2.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of TwoSpin.Speed2.SlowProj from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
          convertLayout(TwoSpinCMat.Y_[p], CoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCMat.A[p], CoarseLFUpstreamTmp);
        }
      }

      // Twospin layout speedlevel 2, fast projects /////////////////////////////

      else if(elem == "Speed2FastProj") {
        TwoSpinAggregation TwoSpinAggs(FGrid_c, FGrid_f, cb, 1); // 1 = use fast projects
        TwoSpinCoarsenedMatrix TwoSpinCMat(*FGrid_c, *FrbGrid_c, 2, isHermitian); // speedLevel = 2
        undoChiralDoubling(UpstreamAggs.subspace);
        for(int i = 0; i < TwoSpinAggs.Subspace().size(); ++i) TwoSpinAggs.Subspace()[i] = UpstreamAggs.subspace[i];
        performChiralDoubling(UpstreamAggs.subspace);

        BenchmarkFunction(TwoSpinCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_f, *LinOp, TwoSpinAggs);
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
