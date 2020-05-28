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
#include <CoarsenedMatrixUpstreamImprovedDirsaveLut.h>
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
  const int  nBasis    = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int  nB        = nBasis / 2;
  Coordinate blockSize = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({4, 4, 4, 4}));
  uint64_t   nIterMin  = readFromCommandLineInt(&argc, &argv, "--miniter", 1000);
  uint64_t   nSecMin   = readFromCommandLineInt(&argc, &argv, "--minsec", 5);
  int        gsPasses  = readFromCommandLineInt(&argc, &argv, "--gspasses", 1);
  int        Ls        = readFromCommandLineInt(&argc, &argv, "--Ls", 1);
  // clang-format on

  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;
  std::cout << GridLogMessage << "Using Ls = " << Ls << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian*         UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian*         UGrid_c   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_c = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_c);
  GridCartesian*         FGrid_f   = nullptr;
  GridRedBlackCartesian* FrbGrid_f = nullptr;
  GridCartesian*         FGrid_c   = nullptr;
  GridRedBlackCartesian* FrbGrid_c = nullptr;

  if(Ls != 1) {
    FGrid_f   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid_f);
    FrbGrid_f = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid_f);
    FGrid_c   = SpaceTimeGrid::makeFiveDimGrid(1, UGrid_c);
    FrbGrid_c = SpaceTimeGrid::makeFiveDimRedBlackGrid(1, UGrid_c);
  } else {
    FGrid_f   = UGrid_f;
    FrbGrid_f = UrbGrid_f;
    FGrid_c   = UGrid_c;
    FrbGrid_c = UrbGrid_c;
  }

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

  typedef Grid::Upstream::Aggregation<vSpinColourVector, vTComplex, nBasis>                   UpstreamAggregation;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis>                   BaselineAggregation;
  typedef Grid::UpstreamImprovedDirsaveLut::Aggregation<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                                  OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                                  TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>                                 FourSpinAggregation;

  typedef Grid::Upstream::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                   UpstreamCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                   BaselineCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLut::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>                                  OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>                                  TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>                                 FourSpinCoarsenedMatrix;

  typedef UpstreamCoarsenedMatrix::CoarseVector           UpstreamCoarseVector;
  typedef BaselineCoarsenedMatrix::CoarseVector           BaselineCoarseVector;
  typedef ImprovedDirsaveLutCoarsenedMatrix::CoarseVector ImprovedDirsaveLutCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField            OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField            TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField           FourSpinCoarseVector;

  typedef UpstreamCoarsenedMatrix::CoarseMatrix           UpstreamCoarseLinkField;
  typedef BaselineCoarsenedMatrix::CoarseMatrix           BaselineCoarseLinkField;
  typedef ImprovedDirsaveLutCoarsenedMatrix::CoarseMatrix ImprovedDirsaveLutCoarsenedMatrixCoarseLinkField;
  typedef OneSpinCoarsenedMatrix::LinkField               OneSpinCoarseLinkField;
  typedef TwoSpinCoarsenedMatrix::LinkField               TwoSpinCoarseLinkField;
  typedef FourSpinCoarsenedMatrix::LinkField              FourSpinCoarseLinkField;

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  const int cb = 0;

  UpstreamAggregation           UpstreamAggs(FGrid_c, FGrid_f, cb);
  BaselineAggregation           BaselineAggs(FGrid_c, FGrid_f, cb);
  ImprovedDirsaveLutAggregation ImprovedDirsaveLutAggs(FGrid_c, FGrid_f, cb);
  TwoSpinAggregation            TwoSpinAggsSlow(FGrid_c, FGrid_f, cb, 0);
  TwoSpinAggregation            TwoSpinAggsFast(FGrid_c, FGrid_f, cb, 1);

  const int checkOrthog = 1;

  // setup vectors once and distribute them to save time
  // (we check agreement of different impls below)

  UpstreamAggs.CreateSubspaceRandom(FPRNG_f);
  for(int i = 0; i < TwoSpinAggsFast.Subspace().size(); ++i)
    TwoSpinAggsFast.Subspace()[i] = UpstreamAggs.subspace[i];

  performChiralDoubling(UpstreamAggs.subspace);
  UpstreamAggs.Orthogonalise(checkOrthog, 1); // 1 gs pass

  for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) {
    BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];
    ImprovedDirsaveLutAggs.subspace[i] = UpstreamAggs.subspace[i];
  }

  TwoSpinAggsFast.Orthogonalise(checkOrthog, 1); // 1 gs pass
  for(int i = 0; i < TwoSpinAggsSlow.Subspace().size(); ++i)
    TwoSpinAggsSlow.Subspace()[i] = TwoSpinAggsFast.Subspace()[i];

  /////////////////////////////////////////////////////////////////////////////
  //             Calculate numbers needed for performance figures            //
  /////////////////////////////////////////////////////////////////////////////

  auto siteElems_f = getSiteElems<LatticeFermion>();
  auto siteElems_c = getSiteElems<UpstreamCoarseVector>();

  std::cout << GridLogDebug << "siteElems_f = " << siteElems_f << std::endl;
  std::cout << GridLogDebug << "siteElems_c = " << siteElems_c << std::endl;

  double UVolume_f = std::accumulate(UGrid_f->_fdimensions.begin(), UGrid_f->_fdimensions.end(), 1, std::multiplies<double>());
  double FVolume_f = std::accumulate(FGrid_f->_fdimensions.begin(), FGrid_f->_fdimensions.end(), 1, std::multiplies<double>());
  double UVolume_c = std::accumulate(UGrid_c->_fdimensions.begin(), UGrid_c->_fdimensions.end(), 1, std::multiplies<double>());
  double FVolume_c = std::accumulate(FGrid_c->_fdimensions.begin(), FGrid_c->_fdimensions.end(), 1, std::multiplies<double>());

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for ProjectToSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    LatticeFermion                 FineVec(FGrid_f);              random(FPRNG_f, FineVec);
    UpstreamCoarseVector           CoarseVecUpstream(FGrid_c);    CoarseVecUpstream                  = Zero();
    BaselineCoarseVector           CoarseVecBaseline(FGrid_c);    CoarseVecBaseline                  = Zero();
    ImprovedDirsaveLutCoarseVector CoarseVecImprovedDirsaveLut(FGrid_c); CoarseVecImprovedDirsaveLut = Zero();
    TwoSpinCoarseVector            CoarseVecTwospinSlow(FGrid_c); CoarseVecTwospinSlow               = Zero();
    TwoSpinCoarseVector            CoarseVecTwospinFast(FGrid_c); CoarseVecTwospinFast               = Zero();
    UpstreamCoarseVector           CoarseVecUpstreamTmp(FGrid_c);
    // clang-format on

    double flop = FVolume_f * (8 * siteElems_f) * nBasis;
    double byte = FVolume_f * (2 * 1 + 2 * siteElems_f) * nBasis * sizeof(Complex);

    BenchmarkFunction(UpstreamAggs.ProjectToSubspace,           flop, byte, nIterMin, nSecMin, CoarseVecUpstream,           FineVec);
    BenchmarkFunction(BaselineAggs.ProjectToSubspace,           flop, byte, nIterMin, nSecMin, CoarseVecBaseline,           FineVec);
    BenchmarkFunction(ImprovedDirsaveLutAggs.ProjectToSubspace, flop, byte, nIterMin, nSecMin, CoarseVecImprovedDirsaveLut, FineVec);
    BenchmarkFunction(TwoSpinAggsSlow.ProjectToSubspace,        flop, byte, nIterMin, nSecMin, CoarseVecTwospinSlow,        FineVec);
    BenchmarkFunction(TwoSpinAggsFast.ProjectToSubspace,        flop, byte, nIterMin, nSecMin, CoarseVecTwospinFast,        FineVec);

    // clang-format off
    printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecBaseline);
    printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecImprovedDirsaveLut);
    convertLayout(CoarseVecTwospinSlow,CoarseVecUpstreamTmp); printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecUpstreamTmp);
    convertLayout(CoarseVecTwospinFast,CoarseVecUpstreamTmp); printDeviationFromReference(tol, CoarseVecUpstream, CoarseVecUpstreamTmp);
    // clang-format on
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for PromoteFromSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    LatticeFermion                 FineVecUpstream(FGrid_f);      FineVecUpstream                    = Zero();
    LatticeFermion                 FineVecBaseline(FGrid_f);      FineVecBaseline                    = Zero();
    LatticeFermion                 FineVecImprovedDirsaveLut(FGrid_f); FineVecImprovedDirsaveLut     = Zero();
    LatticeFermion                 FineVecTwospinSlow(FGrid_f);   FineVecTwospinSlow                 = Zero();
    LatticeFermion                 FineVecTwospinFast(FGrid_f);   FineVecTwospinFast                 = Zero();
    UpstreamCoarseVector           CoarseVecUpstream(FGrid_c);    random(FPRNG_c, CoarseVecUpstream);
    BaselineCoarseVector           CoarseVecBaseline(FGrid_c);    CoarseVecBaseline                  = CoarseVecUpstream;
    ImprovedDirsaveLutCoarseVector CoarseVecImprovedDirsaveLut(FGrid_c); CoarseVecImprovedDirsaveLut = CoarseVecUpstream;
    TwoSpinCoarseVector            CoarseVecTwospinSlow(FGrid_c); convertLayout(CoarseVecUpstream, CoarseVecTwospinSlow);
    TwoSpinCoarseVector            CoarseVecTwospinFast(FGrid_c); convertLayout(CoarseVecUpstream, CoarseVecTwospinFast);
    // clang-format on

    double flop = FVolume_f * (8 * (nBasis - 1) + 6) * siteElems_f;
    double byte = FVolume_f * ((1 * 1 + 3 * siteElems_f) * (nBasis - 1) + (1 * 1 + 2 * siteElems_f) * 1) * sizeof(Complex);

    BenchmarkFunction(UpstreamAggs.PromoteFromSubspace,           flop, byte, nIterMin, nSecMin, CoarseVecUpstream,           FineVecUpstream);
    BenchmarkFunction(BaselineAggs.PromoteFromSubspace,           flop, byte, nIterMin, nSecMin, CoarseVecBaseline,           FineVecBaseline);
    BenchmarkFunction(ImprovedDirsaveLutAggs.PromoteFromSubspace, flop, byte, nIterMin, nSecMin, CoarseVecImprovedDirsaveLut, FineVecImprovedDirsaveLut);
    BenchmarkFunction(TwoSpinAggsSlow.PromoteFromSubspace,        flop, byte, nIterMin, nSecMin, CoarseVecTwospinSlow,        FineVecTwospinSlow);
    BenchmarkFunction(TwoSpinAggsFast.PromoteFromSubspace,        flop, byte, nIterMin, nSecMin, CoarseVecTwospinFast,        FineVecTwospinFast);

    printDeviationFromReference(tol, FineVecUpstream, FineVecBaseline);
    printDeviationFromReference(tol, FineVecUpstream, FineVecImprovedDirsaveLut);
    printDeviationFromReference(tol, FineVecUpstream, FineVecTwospinSlow);
    printDeviationFromReference(tol, FineVecUpstream, FineVecTwospinFast);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Orthogonalise" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    UpstreamAggs.CreateSubspaceRandom(FPRNG_f);
    for(int i=0; i<TwoSpinAggsSlow.Subspace().size(); ++i) {
      TwoSpinAggsSlow.Subspace()[i] = UpstreamAggs.subspace[i];
      TwoSpinAggsFast.Subspace()[i] = UpstreamAggs.subspace[i];
    }
    performChiralDoubling(UpstreamAggs.subspace);
    for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) {
      BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];
      ImprovedDirsaveLutAggs.subspace[i] = UpstreamAggs.subspace[i];
    }

    double flopLocalInnerProduct = FVolume_f * (8 * siteElems_f - 2);
    double byteLocalInnerProduct = FVolume_f * (2 * siteElems_f + 1 * 1) * sizeof(Complex);

    double flopBlockSum = FVolume_f * (2 * 1);
    double byteBlockSum = FVolume_f * (2 * 1 + 1 * 1) * sizeof(Complex);

    double flopCopy = FVolume_c * 0;
    double byteCopy = FVolume_c * (2 * 1) * sizeof(Complex);

    double flopBlockInnerProduct = flopLocalInnerProduct + flopBlockSum + flopCopy;
    double byteBlockInnerProduct = byteLocalInnerProduct + byteBlockSum + byteCopy;

    double flopPow = FVolume_c * (1); // TODO: Put in actual value instead of 1
    double bytePow = FVolume_c * (2 * 1) * sizeof(Complex);

    double flopBlockZAXPY = FVolume_f * (8 * siteElems_f);
    double byteBlockZAXPY = FVolume_f * (3 * siteElems_f + 1 * 1) * sizeof(Complex);

    double flopBlockNormalise = flopBlockInnerProduct + flopPow + flopBlockZAXPY;
    double byteBlockNormalise = byteBlockInnerProduct + bytePow + byteBlockZAXPY;

    double flopMinus = FVolume_c * (6 * 1) * sizeof(Complex);
    double byteMinus = FVolume_c * (2 * 1) * sizeof(Complex);

    double flop = flopBlockNormalise * nBasis + (flopBlockInnerProduct + flopMinus + flopBlockZAXPY) * nBasis * (nBasis - 1) / 2.;
    double byte = byteBlockNormalise * nBasis + (byteBlockInnerProduct + byteMinus + byteBlockZAXPY) * nBasis * (nBasis - 1) / 2.;

    uint64_t nIterOnce = 1;
    uint64_t nSecOnce  = 100;
    auto checkOrthog = 0;

    BenchmarkFunction(UpstreamAggs.Orthogonalise,           flop, byte, nIterOnce, nSecOnce, checkOrthog, gsPasses);
    BenchmarkFunction(BaselineAggs.Orthogonalise,           flop, byte, nIterOnce, nSecOnce, checkOrthog, gsPasses);
    BenchmarkFunction(ImprovedDirsaveLutAggs.Orthogonalise, flop, byte, nIterOnce, nSecOnce, checkOrthog, gsPasses);
    BenchmarkFunction(TwoSpinAggsSlow.Orthogonalise,        flop, byte, nIterOnce, nSecOnce, checkOrthog, gsPasses);
    BenchmarkFunction(TwoSpinAggsFast.Orthogonalise,        flop, byte, nIterOnce, nSecOnce, checkOrthog, gsPasses);

    undoChiralDoubling(UpstreamAggs.subspace); // necessary for comparison
    undoChiralDoubling(BaselineAggs.subspace); // necessary for comparison
    undoChiralDoubling(ImprovedDirsaveLutAggs.subspace); // necessary for comparison

    std::cout << GridLogMessage << "Deviations of BaselineAggs.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], BaselineAggs.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of ImprovedDirsaveLutAggs.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], ImprovedDirsaveLutAggs.subspace[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsSlow.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], TwoSpinAggsSlow.Subspace()[i]);
    std::cout << GridLogMessage << "Deviations of TwoSpinAggsFast.Subspace() from UpstreamAggs.subspace" << std::endl;
    for(auto i = 0; i < nB; ++i) printDeviationFromReference(tol, UpstreamAggs.subspace[i], TwoSpinAggsFast.Subspace()[i]);
  }

  Grid_finalize();
}
