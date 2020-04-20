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
#include <tests/multigrid/Multigrid.h>
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

  typedef Grid::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> OriginalCoarsenedMatrix;
  typedef Grid::CoarsenedMatrixDevelop<vSpinColourVector, vTComplex, nBasis> DevelopCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>      OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>      TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>     FourSpinCoarsenedMatrix;

  typedef OriginalCoarsenedMatrix::CoarseVector OriginalCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField  OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField  TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField FourSpinCoarseVector;

  typedef OriginalCoarsenedMatrix::CoarseMatrix OriginalCoarseLinkField;
  typedef OneSpinCoarsenedMatrix::LinkField     OneSpinCoarseLinkField;
  typedef TwoSpinCoarsenedMatrix::LinkField     TwoSpinCoarseLinkField;
  typedef FourSpinCoarsenedMatrix::LinkField    FourSpinCoarseLinkField;

  typedef Grid::Aggregation<vSpinColourVector, vTComplex, nBasis> OriginalAggregation;
  typedef Grid::AggregationDevelop<vSpinColourVector, vTComplex, nBasis> DevelopAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>      OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>      TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>     FourSpinAggregation;

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  std::cout << GridLogMessage << "Lorentz Index: " << TwoSpinCoarseningPolicy::Nl_f << std::endl;

  OriginalAggregation OriginalAggs(CGrid, FGrid, 0);
  DevelopAggregation  DevelopAggs(CGrid, FGrid, 0);
  TwoSpinAggregation  TwoSpinAggsDefault(CGrid, FGrid, 0, 0);
  TwoSpinAggregation  TwoSpinAggsFast(CGrid, FGrid, 0, 1);

  OriginalAggs.CreateSubspaceRandom(FPRNG);
  for(int i = 0; i < TwoSpinAggsDefault.Subspace().size(); ++i) {
    TwoSpinAggsDefault.Subspace()[i] = OriginalAggs.subspace[i];
    TwoSpinAggsFast.Subspace()[i]    = OriginalAggs.subspace[i];
  }
  for(int i = 0; i < OriginalAggs.subspace.size(); ++i)
    DevelopAggs.subspace[i] = OriginalAggs.subspace[i];
  performChiralDoubling(OriginalAggs.subspace);
  performChiralDoubling(DevelopAggs.Subspace());

  /////////////////////////////////////////////////////////////////////////////
  //             Calculate numbers needed for performance figures            //
  /////////////////////////////////////////////////////////////////////////////

  auto FSiteElems = getSiteElems<LatticeFermion>();
  auto CSiteElems = getSiteElems<OriginalCoarseVector>();

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
    OriginalCoarseVector CoarseVecOriginal(CGrid);        CoarseVecOriginal       = Zero();
    OriginalCoarseVector CoarseVecDevelop(CGrid);         CoarseVecDevelop        = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinDefault(CGrid);  CoarseVecTwospinDefault = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinFast(CGrid);     CoarseVecTwospinFast    = Zero();
    OriginalCoarseVector CoarseVecOriginalTmp(CGrid);
    // clang-format on

    double flop = FVolume * (8 * FSiteElems) * nBasis;
    double byte = FVolume * (2 * 1 + 2 * FSiteElems) * nBasis * sizeof(Complex);

    if(doPerfProfiling) {
      PerfProfileFunction(OriginalAggs.ProjectToSubspace,       nIter, CoarseVecOriginal,       FineVec);
      PerfProfileFunction(DevelopAggs.ProjectToSubspace,        nIter, CoarseVecDevelop,        FineVec);
      PerfProfileFunction(TwoSpinAggsFast.ProjectToSubspace,    nIter, CoarseVecTwospinFast,    FineVec);
    } else {
      BenchmarkFunction(OriginalAggs.ProjectToSubspace,       flop, byte, nIter, CoarseVecOriginal,       FineVec);
      BenchmarkFunction(DevelopAggs.ProjectToSubspace,        flop, byte, nIter, CoarseVecDevelop,        FineVec);
      BenchmarkFunction(TwoSpinAggsFast.ProjectToSubspace,    flop, byte, nIter, CoarseVecTwospinFast,    FineVec);
    }

    prettyPrintProfiling("", TwoSpinAggsDefault.GetProfile(), GridTime(0), true);
    prettyPrintProfiling("", TwoSpinAggsFast.GetProfile(),    GridTime(0), true);

    // clang-format off
    printDeviationFromReference(tol, CoarseVecOriginal, CoarseVecDevelop);
    convertLayout(CoarseVecTwospinFast,    CoarseVecOriginalTmp); printDeviationFromReference(tol, CoarseVecOriginal, CoarseVecOriginalTmp);
    // clang-format on
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for PromoteFromSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    LatticeFermion        FineVecOriginal(FGrid);         FineVecOriginal       = Zero();
    LatticeFermion        FineVecDevelop(FGrid);          FineVecDevelop        = Zero();
    LatticeFermion        FineVecTwospinDefault(FGrid);   FineVecTwospinDefault = Zero();
    LatticeFermion        FineVecTwospinFast(FGrid);      FineVecTwospinFast    = Zero();
    OriginalCoarseVector  CoarseVecOriginal(CGrid);       random(CPRNG, CoarseVecOriginal);
    OriginalCoarseVector  CoarseVecDevelop(CGrid);        CoarseVecDevelop = CoarseVecOriginal;
    TwoSpinCoarseVector   CoarseVecTwospinDefault(CGrid); convertLayout(CoarseVecOriginal, CoarseVecTwospinDefault);
    TwoSpinCoarseVector   CoarseVecTwospinFast(CGrid);    convertLayout(CoarseVecOriginal, CoarseVecTwospinFast);
    // clang-format on

    double flop = FVolume * (8 * (nBasis - 1) + 6) * FSiteElems;
    double byte = FVolume * ((1 * 1 + 3 * FSiteElems) * (nBasis - 1) + (1 * 1 + 2 * FSiteElems) * 1) * sizeof(Complex);

    if(doPerfProfiling) {
      PerfProfileFunction(OriginalAggs.PromoteFromSubspace,       nIter, CoarseVecOriginal,       FineVecOriginal);
      PerfProfileFunction(DevelopAggs.PromoteFromSubspace,        nIter, CoarseVecDevelop,        FineVecDevelop);
      PerfProfileFunction(TwoSpinAggsFast.PromoteFromSubspace,    nIter, CoarseVecTwospinFast,    FineVecTwospinFast);
    } else {
      BenchmarkFunction(OriginalAggs.PromoteFromSubspace,       flop, byte, nIter, CoarseVecOriginal,       FineVecOriginal);
      BenchmarkFunction(DevelopAggs.PromoteFromSubspace,        flop, byte, nIter, CoarseVecDevelop,        FineVecDevelop);
      BenchmarkFunction(TwoSpinAggsFast.PromoteFromSubspace,    flop, byte, nIter, CoarseVecTwospinFast,    FineVecTwospinFast);
    }

    printDeviationFromReference(tol, FineVecOriginal, FineVecDevelop);
    printDeviationFromReference(tol, FineVecOriginal, FineVecTwospinFast);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Orthogonalise" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    auto nIterOne = 1;

    OriginalAggs.CreateSubspaceRandom(FPRNG);
    for(int i=0; i<TwoSpinAggsDefault.Subspace().size(); ++i) {
      TwoSpinAggsDefault.Subspace()[i] = OriginalAggs.subspace[i];
      TwoSpinAggsFast.Subspace()[i]    = OriginalAggs.subspace[i];
    }
    for(int i = 0; i < OriginalAggs.subspace.size(); ++i)
      DevelopAggs.subspace[i] = OriginalAggs.subspace[i];
    performChiralDoubling(OriginalAggs.subspace);
    performChiralDoubling(DevelopAggs.Subspace());

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
      PerfProfileFunction(DevelopAggs.Orthogonalise,        nIterOne, 1); // 1 pass
      PerfProfileFunction(OriginalAggs.Orthogonalise,       nIterOne);
      PerfProfileFunction(TwoSpinAggsDefault.Orthogonalise, nIterOne, 0, 1); // no orthog check, 1 pass
      PerfProfileFunction(TwoSpinAggsFast.Orthogonalise,    nIterOne, 0, 1); // no orthog check, 1 pass
    } else {
      BenchmarkFunction(DevelopAggs.Orthogonalise,        flop, byte, nIterOne, 1); // 1 pass
      BenchmarkFunction(OriginalAggs.Orthogonalise,       flop, byte, nIterOne);
      BenchmarkFunction(TwoSpinAggsDefault.Orthogonalise, flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
      BenchmarkFunction(TwoSpinAggsFast.Orthogonalise,    flop, byte, nIterOne, 0, 1); // no orthog check, 1 pass
    }

    undoChiralDoubling(OriginalAggs.subspace);  // necessary for comparison
    undoChiralDoubling(DevelopAggs.Subspace()); // necessary for comparison

    std::cout << GridLogMessage << "Deviations of DevelopAggs.Subspace() from OriginalAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, OriginalAggs.subspace[i], DevelopAggs.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsDefault.Subspace() from OriginalAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, OriginalAggs.subspace[i], TwoSpinAggsDefault.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsFast.Subspace() from OriginalAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, OriginalAggs.subspace[i], TwoSpinAggsFast.Subspace()[i]);
  }

  Grid_finalize();
}
