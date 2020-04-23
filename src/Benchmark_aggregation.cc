/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_aggregation.cc

    Copyright (C) 2015 - 2018

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
#include <Benchmark_helpers.h>
#include <Layout_converters.h>

using namespace Grid;
using namespace Grid::Rework;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::LayoutConverters;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_aggregation
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  const int  nBasis          = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int  nB              = nBasis / 2;
  Coordinate blockSize       = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({4, 4, 4, 4}));
  int        nIter           = readFromCommandLineInt(&argc, &argv, "--niter", 10);
  bool       doPerfProfiling = readFromCommandLineToggle(&argc, &argv, "--perfprofiling");
  // clang-format on

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
#else // 4d use case (f = fine, c = coarse)
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
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningPolicy<LatticeFermion, nB, 1> OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2> TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4> FourSpinCoarseningPolicy;

  typedef Grid::Upstream::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> UpstreamCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> BaselineCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>                OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>                TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>               FourSpinCoarsenedMatrix;

  typedef UpstreamCoarsenedMatrix::CoarseVector UpstreamCoarseVector;
  typedef BaselineCoarsenedMatrix::CoarseVector BaselineCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField  OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField  TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField FourSpinCoarseVector;

  typedef UpstreamCoarsenedMatrix::CoarseMatrix UpstreamCoarseLinkField;
  typedef BaselineCoarsenedMatrix::CoarseMatrix BaselineCoarseLinkField;
  typedef OneSpinCoarsenedMatrix::LinkField     OneSpinCoarseLinkField;
  typedef TwoSpinCoarsenedMatrix::LinkField     TwoSpinCoarseLinkField;
  typedef FourSpinCoarsenedMatrix::LinkField    FourSpinCoarseLinkField;

  typedef Grid::Upstream::Aggregation<vSpinColourVector, vTComplex, nBasis> UpstreamAggregation;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis> BaselineAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>               FourSpinAggregation;

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  std::cout << GridLogMessage << "Lorentz Index: " << TwoSpinCoarseningPolicy::Nl_f << std::endl;

  UpstreamAggregation UpstreamAggs(CGrid, FGrid, 0);
  BaselineAggregation BaselineAggs(CGrid, FGrid, 0);
  TwoSpinAggregation  TwoSpinAggsDefault(CGrid, FGrid, 0, 0);
  TwoSpinAggregation  TwoSpinAggsFast(CGrid, FGrid, 0, 1);

  UpstreamAggs.CreateSubspaceRandom(FPRNG);
  for(int i = 0; i < TwoSpinAggsDefault.Subspace().size(); ++i) {
    TwoSpinAggsDefault.Subspace()[i] = UpstreamAggs.subspace[i];
    TwoSpinAggsFast.Subspace()[i]    = UpstreamAggs.subspace[i];
  }
  for(int i = 0; i < UpstreamAggs.subspace.size(); ++i)
    BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];
  performChiralDoubling(UpstreamAggs.subspace);
  performChiralDoubling(BaselineAggs.subspace);

  /////////////////////////////////////////////////////////////////////////////
  //             Calculate numbers needed for performance figures            //
  /////////////////////////////////////////////////////////////////////////////

  auto FSiteElems = getSiteElems<LatticeFermion>();
  auto CSiteElems = getSiteElems<UpstreamCoarseVector>();

  double FVolume = std::accumulate(FGrid->_fdimensions.begin(), FGrid->_fdimensions.end(), 1, std::multiplies<double>());
  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for ProjectToSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    LatticeFermion       FineVec(FGrid);                  random(FPRNG, FineVec);
    UpstreamCoarseVector CoarseVecUpstream(CGrid);        CoarseVecUpstream       = Zero();
    BaselineCoarseVector CoarseVecBaseline(CGrid);        CoarseVecBaseline       = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinDefault(CGrid);  CoarseVecTwospinDefault = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinFast(CGrid);     CoarseVecTwospinFast    = Zero();
    UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
    // clang-format on

    double flop = FVolume * (8 * FSiteElems) * nBasis;
    double byte = FVolume * (2 * 1 + 2 * FSiteElems) * nBasis * sizeof(Complex);

    if(doPerfProfiling) {
      PerfProfileFunction(UpstreamAggs.ProjectToSubspace,       nIter, CoarseVecUpstream,       FineVec);
      PerfProfileFunction(BaselineAggs.ProjectToSubspace,       nIter, CoarseVecBaseline,       FineVec);
      PerfProfileFunction(TwoSpinAggsDefault.ProjectToSubspace, nIter, CoarseVecTwospinDefault, FineVec);
      PerfProfileFunction(TwoSpinAggsFast.ProjectToSubspace,    nIter, CoarseVecTwospinFast,    FineVec);
    } else {
      BenchmarkFunction(UpstreamAggs.ProjectToSubspace,       flop, byte, nIter, CoarseVecUpstream,       FineVec);
      BenchmarkFunction(BaselineAggs.ProjectToSubspace,       flop, byte, nIter, CoarseVecBaseline,       FineVec);
      BenchmarkFunction(TwoSpinAggsDefault.ProjectToSubspace, flop, byte, nIter, CoarseVecTwospinDefault, FineVec);
      BenchmarkFunction(TwoSpinAggsFast.ProjectToSubspace,    flop, byte, nIter, CoarseVecTwospinFast,    FineVec);
    }

    // prettyPrintProfiling("", TwoSpinAggsDefault.GetProfile(), GridTime(0), true);
    // prettyPrintProfiling("", TwoSpinAggsFast.GetProfile(),    GridTime(0), true);

    // clang-format off
    printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecBaseline);
    convertLayout(CoarseVecTwospinFast,CoarseVecUpstreamTmp); printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecUpstreamTmp);
    // clang-format on
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for PromoteFromSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    LatticeFermion        FineVecUpstream(FGrid);         FineVecUpstream       = Zero();
    LatticeFermion        FineVecBaseline(FGrid);         FineVecBaseline       = Zero();
    LatticeFermion        FineVecTwospinDefault(FGrid);   FineVecTwospinDefault = Zero();
    LatticeFermion        FineVecTwospinFast(FGrid);      FineVecTwospinFast    = Zero();
    UpstreamCoarseVector  CoarseVecUpstream(CGrid);       random(CPRNG, CoarseVecUpstream);
    BaselineCoarseVector  CoarseVecBaseline(CGrid);       CoarseVecBaseline = CoarseVecUpstream;
    TwoSpinCoarseVector   CoarseVecTwospinDefault(CGrid); convertLayout(CoarseVecUpstream, CoarseVecTwospinDefault);
    TwoSpinCoarseVector   CoarseVecTwospinFast(CGrid);    convertLayout(CoarseVecUpstream, CoarseVecTwospinFast);
    // clang-format on

    double flop = FVolume * (8 * (nBasis - 1) + 6) * FSiteElems;
    double byte = FVolume * ((1 * 1 + 3 * FSiteElems) * (nBasis - 1) + (1 * 1 + 2 * FSiteElems) * 1) * sizeof(Complex);

    if(doPerfProfiling) {
      PerfProfileFunction(UpstreamAggs.PromoteFromSubspace,       nIter, CoarseVecUpstream,       FineVecUpstream);
      PerfProfileFunction(BaselineAggs.PromoteFromSubspace,       nIter, CoarseVecBaseline,       FineVecBaseline);
      PerfProfileFunction(TwoSpinAggsDefault.PromoteFromSubspace, nIter, CoarseVecTwospinDefault, FineVecTwospinDefault);
      PerfProfileFunction(TwoSpinAggsFast.PromoteFromSubspace,    nIter, CoarseVecTwospinFast,    FineVecTwospinFast);
    } else {
      BenchmarkFunction(UpstreamAggs.PromoteFromSubspace,       flop, byte, nIter, CoarseVecUpstream,       FineVecUpstream);
      BenchmarkFunction(BaselineAggs.PromoteFromSubspace,       flop, byte, nIter, CoarseVecBaseline,       FineVecBaseline);
      BenchmarkFunction(TwoSpinAggsDefault.PromoteFromSubspace, flop, byte, nIter, CoarseVecTwospinDefault, FineVecTwospinDefault);
      BenchmarkFunction(TwoSpinAggsFast.PromoteFromSubspace,    flop, byte, nIter, CoarseVecTwospinFast,    FineVecTwospinFast);
    }

    printDeviationFromReference(tol, FineVecUpstream, FineVecBaseline);
    printDeviationFromReference(tol, FineVecUpstream, FineVecTwospinDefault);
    printDeviationFromReference(tol, FineVecUpstream, FineVecTwospinFast);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Orthogonalise" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    auto nIterOne = 1;

    UpstreamAggs.CreateSubspaceRandom(FPRNG);
    for(int i=0; i<TwoSpinAggsDefault.Subspace().size(); ++i) {
      TwoSpinAggsDefault.Subspace()[i] = UpstreamAggs.subspace[i];
      TwoSpinAggsFast.Subspace()[i]    = UpstreamAggs.subspace[i];
    }
    for(int i = 0; i < UpstreamAggs.subspace.size(); ++i)
      BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];
    performChiralDoubling(UpstreamAggs.subspace);
    performChiralDoubling(BaselineAggs.subspace);

    double flopLocalInnerProduct = FVolume * (8 * FSiteElems - 2);
    double byteLocalInnerProduct = FVolume * (2 * FSiteElems + 1 * 1) * sizeof(Complex);

    double flopBlockSum = FVolume * (2 * 1);
    double byteBlockSum = FVolume * (2 * 1 + 1 * 1) * sizeof(Complex);

    double flopCopy = CVolume * 0;
    double byteCopy = CVolume * (2 * 1) * sizeof(Complex);

    double flopBlockInnerProduct = flopLocalInnerProduct + flopBlockSum + flopCopy;
    double byteBlockInnerProduct = byteLocalInnerProduct + byteBlockSum + byteCopy;

    double flopPow = CVolume * (1); // TODO: Put in actual value instead of 1
    double bytePow = CVolume * (2 * 1) * sizeof(Complex);

    double flopBlockZAXPY = FVolume * (8 * FSiteElems);
    double byteBlockZAXPY = FVolume * (3 * FSiteElems + 1 * 1) * sizeof(Complex);

    double flopBlockNormalise = flopBlockInnerProduct + flopPow + flopBlockZAXPY;
    double byteBlockNormalise = byteBlockInnerProduct + bytePow + byteBlockZAXPY;

    double flopMinus = CVolume * (6 * 1) * sizeof(Complex);
    double byteMinus = CVolume * (2 * 1) * sizeof(Complex);

    double flop = flopBlockNormalise * nBasis + (flopBlockInnerProduct + flopMinus + flopBlockZAXPY) * nBasis * (nBasis - 1) / 2.;
    double byte = byteBlockNormalise * nBasis + (byteBlockInnerProduct + byteMinus + byteBlockZAXPY) * nBasis * (nBasis - 1) / 2.;

    if(doPerfProfiling) {
      PerfProfileFunction(BaselineAggs.Orthogonalise,       nIterOne, 0, 1); // no orthog check, 1 pass
      PerfProfileFunction(UpstreamAggs.Orthogonalise,       nIterOne, 0, 1); // no orthog check, 1 pass
      PerfProfileFunction(TwoSpinAggsDefault.Orthogonalise, nIterOne, 0, 1); // no orthog check, 1 pass
      PerfProfileFunction(TwoSpinAggsFast.Orthogonalise,    nIterOne, 0, 1); // no orthog check, 1 pass
    } else {
      BenchmarkFunction(BaselineAggs.Orthogonalise,       flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
      BenchmarkFunction(UpstreamAggs.Orthogonalise,       flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
      BenchmarkFunction(TwoSpinAggsDefault.Orthogonalise, flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
      BenchmarkFunction(TwoSpinAggsFast.Orthogonalise,    flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
    }

    undoChiralDoubling(UpstreamAggs.subspace); // necessary for comparison
    undoChiralDoubling(BaselineAggs.subspace); // necessary for comparison

    std::cout << GridLogMessage << "Deviations of BaselineAggs.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], BaselineAggs.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsDefault.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], TwoSpinAggsDefault.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsFast.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], TwoSpinAggsFast.Subspace()[i]);
  }

  Grid_finalize();
}
