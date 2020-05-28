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

  Coordinate clatt  = GridDefaultLatt();
  Coordinate cclatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian*         UGrid_c    = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_c  = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_c);
  GridCartesian*         UGrid_cc   = SpaceTimeGrid::makeFourDimGrid(cclatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_cc = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_cc);
  GridCartesian*         FGrid_c    = UGrid_c;
  GridRedBlackCartesian* FrbGrid_c  = UrbGrid_c;
  GridCartesian*         FGrid_cc   = UGrid_cc;
  GridRedBlackCartesian* FrbGrid_cc = UrbGrid_cc;

  UGrid_c->show_decomposition();
  FGrid_c->show_decomposition();
  UGrid_cc->show_decomposition();
  FGrid_cc->show_decomposition();

  GridParallelRNG UPRNG_c(UGrid_c);
  GridParallelRNG FPRNG_c(FGrid_c);
  GridParallelRNG UPRNG_cc(UGrid_cc);
  GridParallelRNG FPRNG_cc(FGrid_cc);

  std::vector<int> seeds({1, 2, 3, 4});

  UPRNG_c.SeedFixedIntegers(seeds);
  FPRNG_c.SeedFixedIntegers(seeds);
  UPRNG_cc.SeedFixedIntegers(seeds);
  FPRNG_cc.SeedFixedIntegers(seeds);

  RealD tol = getPrecision<vComplex>::value == 2 ? 1e-15 : 1e-7;

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  // coarse level /////////////////////////////////////////////////////////////

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

  // coarse coarse level //////////////////////////////////////////////////////

  typedef CoarseningPolicy<OneSpinCoarseVector, nB, 1>  OneSpinCoarseCoarseningPolicy;
  typedef CoarseningPolicy<TwoSpinCoarseVector, nB, 2>  TwoSpinCoarseCoarseningPolicy;
  typedef CoarseningPolicy<FourSpinCoarseVector, nB, 4> FourSpinCoarseCoarseningPolicy;

  typedef Grid::Upstream::Aggregation<UpstreamAggregation::siteVector, iScalar<vTComplex>, nBasis>                                     UpstreamCoarseAggregation;
  typedef Grid::Baseline::Aggregation<BaselineAggregation::siteVector, iScalar<vTComplex>, nBasis>                                     BaselineCoarseAggregation;
  typedef Grid::UpstreamImprovedDirsave::Aggregation<ImprovedDirsaveAggregation::siteVector, iScalar<vTComplex>, nBasis>               ImprovedDirsaveCoarseAggregation;
  typedef Grid::UpstreamImprovedDirsaveLut::Aggregation<ImprovedDirsaveLutAggregation::siteVector, iScalar<vTComplex>, nBasis>         ImprovedDirsaveLutCoarseAggregation;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::Aggregation<ImprovedDirsaveLutMRHSAggregation::siteVector, iScalar<vTComplex>, nBasis> ImprovedDirsaveLutMRHSCoarseAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseCoarseningPolicy>                                                                     OneSpinCoarseAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseCoarseningPolicy>                                                                     TwoSpinCoarseAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseCoarseningPolicy>                                                                    FourSpinCoarseAggregation;

  typedef Grid::Upstream::CoarsenedMatrix<UpstreamCoarsenedMatrix::siteVector, iScalar<vTComplex>, nBasis>                                     UpstreamCoarseCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<BaselineCoarsenedMatrix::siteVector, iScalar<vTComplex>, nBasis>                                     BaselineCoarseCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsave::CoarsenedMatrix<ImprovedDirsaveCoarsenedMatrix::siteVector, iScalar<vTComplex>, nBasis>               ImprovedDirsaveCoarseCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLut::CoarsenedMatrix<ImprovedDirsaveLutCoarsenedMatrix::siteVector, iScalar<vTComplex>, nBasis>         ImprovedDirsaveLutCoarseCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::CoarsenedMatrix<ImprovedDirsaveLutMRHSCoarsenedMatrix::siteVector, iScalar<vTComplex>, nBasis> ImprovedDirsaveLutMRHSCoarseCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseCoarseningPolicy>                                                                         OneSpinCoarseCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseCoarseningPolicy>                                                                         TwoSpinCoarseCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseCoarseningPolicy>                                                                        FourSpinCoarseCoarsenedMatrix;

  typedef UpstreamCoarseCoarsenedMatrix::CoarseVector               UpstreamCoarseCoarseVector;
  typedef BaselineCoarseCoarsenedMatrix::CoarseVector               BaselineCoarseCoarseVector;
  typedef ImprovedDirsaveCoarseCoarsenedMatrix::CoarseVector        ImprovedDirsaveCoarseCoarseVector;
  typedef ImprovedDirsaveLutCoarseCoarsenedMatrix::CoarseVector     ImprovedDirsaveLutCoarseCoarseVector;
  typedef ImprovedDirsaveLutMRHSCoarseCoarsenedMatrix::CoarseVector ImprovedDirsaveLutMRHSCoarseCoarseVector;
  typedef OneSpinCoarseCoarsenedMatrix::FermionField                OneSpinCoarseCoarseVector;
  typedef TwoSpinCoarseCoarsenedMatrix::FermionField                TwoSpinCoarseCoarseVector;
  typedef FourSpinCoarseCoarsenedMatrix::FermionField               FourSpinCoarseCoarseVector;

  typedef UpstreamCoarseCoarsenedMatrix::CoarseMatrix               UpstreamCoarseCoarseLinkField;
  typedef BaselineCoarseCoarsenedMatrix::CoarseMatrix               BaselineCoarseCoarseLinkField;
  typedef ImprovedDirsaveCoarseCoarsenedMatrix::CoarseMatrix        ImprovedDirsaveCoarseCoarseLinkField;
  typedef ImprovedDirsaveLutCoarseCoarsenedMatrix::CoarseMatrix     ImprovedDirsaveLutCoarseCoarseLinkField;
  typedef ImprovedDirsaveLutMRHSCoarseCoarsenedMatrix::CoarseMatrix ImprovedDirsaveLutMRHSCoarseCoarseLinkField;
  typedef OneSpinCoarseCoarsenedMatrix::LinkField                   OneSpinCoarseCoarseLinkField;
  typedef TwoSpinCoarseCoarsenedMatrix::LinkField                   TwoSpinCoarseCoarseLinkField;
  typedef FourSpinCoarseCoarsenedMatrix::LinkField                  FourSpinCoarseCoarseLinkField;

  /////////////////////////////////////////////////////////////////////////////
  //                       Set values for some toggles                       //
  /////////////////////////////////////////////////////////////////////////////

  const int cb          = 0; // cb to use in aggregation
  const int checkOrthog = 1; // whether to check orthog in setup of aggregation
  const int gsPasses    = 1; // number of GS in setup of aggregation
  const int isHermitian = 0; // whether we do Petrov-Galerkin (hermitian) or Galerkin (G5-hermitian) coarsening

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarsenedMatrix UpstreamCMat(*FGrid_c, isHermitian);
  for(int p : {4, 5, 6, 7, 8}) // randomize only self and backwards links to have correct relation between forward and backward
    random(FPRNG_c, UpstreamCMat.A[p]);
  if(isHermitian)
    UpstreamCMat.ForceHermitian();
  else
    UpstreamCMat.ConstructRemainingLinks();

  MdagMLinearOperator<UpstreamCoarsenedMatrix, UpstreamCoarseVector> LinOp(UpstreamCMat);

  GridCartesian*         FGrid_c_5d    = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, FGrid_c);
  GridRedBlackCartesian* FrbGrid_c_5d  = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, FGrid_c);
  GridCartesian*         FGrid_cc_5d   = SpaceTimeGrid::makeFiveDimGrid(nParSetupVecs, FGrid_cc);
  GridRedBlackCartesian* FrbGrid_cc_5d = SpaceTimeGrid::makeFiveDimRedBlackGrid(nParSetupVecs, FGrid_cc);

  // WilsonMRHSFermionR       Dw5(Umu, *FGrid_c_5d, *FrbGrid_c_5d, *FGrid_c, *FrbGrid_c, mass);
  // WilsonCloverMRHSFermionR Dwc5(Umu, *FGrid_c_5d, *FrbGrid_c_5d, *FGrid_c, *FrbGrid_c, mass, csw, csw);

  // MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion> LinOpDw5(Dw5);
  // MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion> LinOpDwc5(Dwc5);

  // MdagMLinearOperator<WilsonMRHSFermionR, LatticeFermion>* LinOp5;

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarseAggregation UpstreamCoarseAggs(FGrid_cc, FGrid_c, cb);

  UpstreamCoarseAggs.CreateSubspaceRandom(FPRNG_c);
  performChiralDoubling(UpstreamCoarseAggs.subspace);
  UpstreamCoarseAggs.Orthogonalise(checkOrthog, gsPasses);

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarseCoarsenedMatrix UpstreamCoarseCMat(*FGrid_cc, isHermitian);

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil   = UpstreamCoarseCMat.geom.npoint;
  double nAccum     = nStencil - 1;
  double siteElems_c = getSiteElems<UpstreamCoarseVector>();
  double siteElems_cc = getSiteElems<UpstreamCoarseCoarseVector>();

  std::cout << GridLogDebug << "siteElems_c = " << siteElems_c << std::endl;
  std::cout << GridLogDebug << "siteElems_cc = " << siteElems_cc << std::endl;

  double FVolume_c = std::accumulate(FGrid_c->_fdimensions.begin(), FGrid_c->_fdimensions.end(), 1, std::multiplies<double>());
  double FVolume_cc = std::accumulate(FGrid_cc->_fdimensions.begin(), FGrid_cc->_fdimensions.end(), 1, std::multiplies<double>());

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

    UpstreamCoarseCoarseLinkField CoarseCoarseLFUpstreamTmp(FGrid_cc);

    double flop = 0; // TODO
    double byte = 0; // TODO

    // Upstream = Reference (always run) //////////////////////////////////////

    BenchmarkFunction(UpstreamCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, UpstreamCoarseAggs);
    auto profResults = UpstreamCoarseCMat.GetProfile(); UpstreamCoarseCMat.ResetProfile();
    prettyPrintProfiling("Upstream", profResults, profResults["CoarsenOperator.Total"].t, false);

    for(auto const& elem : toRun) {
      // Baseline = state when I started working ////////////////////////////////

      std::cout << "Running benchmark for configuration " << elem << std::endl;

      if(elem == "Baseline") {
        BaselineCoarseAggregation BaselineCoarseAggs(FGrid_cc, FGrid_c, cb);
        BaselineCoarseCoarsenedMatrix BaselineCoarseCMat(*FGrid_cc, *FrbGrid_cc, isHermitian);
        for(int i = 0; i < UpstreamCoarseAggs.subspace.size(); ++i) BaselineCoarseAggs.subspace[i] = UpstreamCoarseAggs.subspace[i];

        BenchmarkFunction(BaselineCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, BaselineCoarseAggs);
        profResults = BaselineCoarseCMat.GetProfile(); BaselineCoarseCMat.ResetProfile();
        prettyPrintProfiling("Baseline", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of Baseline from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], BaselineCoarseCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving /////////////////////////////

      else if(elem == "ImprovedDirsave") {
        ImprovedDirsaveCoarseAggregation ImprovedDirsaveCoarseAggs(FGrid_cc, FGrid_c, cb);
        ImprovedDirsaveCoarseCoarsenedMatrix ImprovedDirsaveCoarseCMat(*FGrid_cc, isHermitian);
        for(int i = 0; i < UpstreamCoarseAggs.subspace.size(); ++i) ImprovedDirsaveCoarseAggs.subspace[i] = UpstreamCoarseAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, ImprovedDirsaveCoarseAggs);
        profResults = ImprovedDirsaveCoarseCMat.GetProfile(); ImprovedDirsaveCoarseCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsave", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsave from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], ImprovedDirsaveCoarseCMat.A[p]);
        }
      }

      // Improvements to upstream: direction saving + lut ///////////////////////

      else if(elem == "ImprovedDirsaveLut") {
        ImprovedDirsaveLutCoarseAggregation ImprovedDirsaveLutCoarseAggs(FGrid_cc, FGrid_c, cb);
        ImprovedDirsaveLutCoarseCoarsenedMatrix ImprovedDirsaveLutCoarseCMat(*FGrid_cc, isHermitian);
        for(int i = 0; i < UpstreamCoarseAggs.subspace.size(); ++i) ImprovedDirsaveLutCoarseAggs.subspace[i] = UpstreamCoarseAggs.subspace[i];

        BenchmarkFunction(ImprovedDirsaveLutCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, ImprovedDirsaveLutCoarseAggs);
        profResults = ImprovedDirsaveLutCoarseCMat.GetProfile(); ImprovedDirsaveLutCoarseCMat.ResetProfile();
        prettyPrintProfiling("ImprovedDirsaveLut", profResults, profResults["CoarsenOperator.Total"].t, false);

        std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLut from Upstream" << std::endl;
        for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
          printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], ImprovedDirsaveLutCoarseCMat.A[p]);
        }
      }

      // My improvements to upstream with MRHS //////////////////////////////////

      // else if(elem == "ImprovedDirsaveLutMRHS") {
      //   ImprovedDirsaveLutMRHSCoarseAggregation ImprovedDirsaveLutMRHSCoarseAggs(FGrid_cc, FGrid_c, cb);
      //   ImprovedDirsaveLutMRHSCoarseCoarsenedMatrix ImprovedDirsaveLutMRHSCoarseCMat(*FGrid_c_5d, *FGrid_c, *FGrid_cc_5d, *FGrid_cc, isHermitian);
      //   for(int i = 0; i < UpstreamCoarseAggs.subspace.size(); ++i) ImprovedDirsaveLutMRHSCoarseAggs.subspace[i] = UpstreamCoarseAggs.subspace[i];

      //   BenchmarkFunction(ImprovedDirsaveLutMRHSCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c_5d, LinOp5, ImprovedDirsaveLutMRHSCoarseAggs);
      //   profResults = ImprovedDirsaveLutMRHSCoarseCMat.GetProfile(); ImprovedDirsaveLutMRHSCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("ImprovedDirsaveLutMRHS", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLutMRHS from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], ImprovedDirsaveLutMRHSCoarseCMat.A[p]);
      //   }
      // }

      // Twospin layout speedlevel 0, slow projects /////////////////////////////

      // else if(elem == "Speed0SlowProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 0); // 0 = don't use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 0, isHermitian); // speedLevel = 0
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed0.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.SlowProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }

      // // Twospin layout speedlevel 0, fast projects /////////////////////////////

      // else if(elem == "Speed0FastProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 1); // 1 = use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 0, isHermitian); // speedLevel = 0
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed0.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed0.FastProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }

      // // Twospin layout speedlevel 1, slow projects /////////////////////////////

      // else if(elem == "Speed1SlowProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 0); // 0 = don't use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 1, isHermitian); // speedLevel = 1
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed1.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.SlowProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }

      // // Twospin layout speedlevel 1, fast projects /////////////////////////////

      // else if(elem == "Speed1FastProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 1); // 1 = use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 1, isHermitian); // speedLevel = 1
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed1.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed1.FastProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }

      // // Twospin layout speedlevel 2, slow projects /////////////////////////////

      // else if(elem == "Speed2SlowProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 0); // 0 = don't use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 2, isHermitian); // speedLevel = 2
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed2.SlowProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed2.SlowProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }

      // // Twospin layout speedlevel 2, fast projects /////////////////////////////

      // else if(elem == "Speed2FastProj") {
      //   TwoSpinCoarseAggregation TwoSpinCoarseAggs(FGrid_cc, FGrid_c, cb, 1); // 1 = use fast projects
      //   TwoSpinCoarseCoarsenedMatrix TwoSpinCoarseCMat(*FGrid_cc, *FrbGrid_cc, 2, isHermitian); // speedLevel = 2
      //   undoChiralDoubling(UpstreamCoarseAggs.subspace);
      //   for(int i = 0; i < TwoSpinCoarseAggs.Subspace().size(); ++i) TwoSpinCoarseAggs.Subspace()[i] = UpstreamCoarseAggs.subspace[i];
      //   performChiralDoubling(UpstreamCoarseAggs.subspace);

      //   BenchmarkFunction(TwoSpinCoarseCMat.CoarsenOperator, flop, byte, nIterOnce, nSecOnce, FGrid_c, LinOp, TwoSpinCoarseAggs);
      //   profResults = TwoSpinCoarseCMat.GetProfile(); TwoSpinCoarseCMat.ResetProfile();
      //   prettyPrintProfiling("TwoSpin.Speed2.FastProj", profResults, profResults["CoarsenOperator.Total"].t, false);

      //   std::cout << GridLogMessage << "Deviations of TwoSpin.Speed2.FastProj from Upstream" << std::endl;
      //   for(int p = 0; p < UpstreamCoarseCMat.geom.npoint; ++p) {
      //     convertLayout(TwoSpinCoarseCMat.Y_[p], CoarseCoarseLFUpstreamTmp); printDeviationFromReference(tol, UpstreamCoarseCMat.A[p], CoarseCoarseLFUpstreamTmp);
      //   }
      // }
    }
  }

  Grid_finalize();
}
