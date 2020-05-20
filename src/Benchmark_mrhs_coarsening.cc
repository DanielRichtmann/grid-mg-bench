/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_mrhs_coarsening.cc

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
#include <CoarsenedMatrixUpstreamImprovedDirsave.h>
#include <CoarsenedMatrixUpstreamImprovedDirsaveLut.h>
#include <Benchmark_helpers.h>
#include <Layout_converters.h>

using namespace Grid;
using namespace Grid::Rework;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::LayoutConverters;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_mrhs_coarsening
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
  int        gsPasses  = readFromCommandLineInt(&argc, &argv, "--gspasses", 1);
  int        nrhs      = readFromCommandLineInt(&argc, &argv, "--nrhs", 20);
  uint64_t   nIterMin  = readFromCommandLineInt(&argc, &argv, "--miniter", 1000);
  uint64_t   nSecMin   = readFromCommandLineInt(&argc, &argv, "--minsec", 5);
  // clang-format on

  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

#if 1 // 5d use case (u = gauge, f = fermion = fine, t = tmp, c = coarse)
  GridCartesian*         UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian*         FGrid_f   = SpaceTimeGrid::makeFiveDimGrid(nrhs, UGrid_f);
  GridRedBlackCartesian* FrbGrid_f = SpaceTimeGrid::makeFiveDimRedBlackGrid(nrhs, UGrid_f);
  GridCartesian*         UGrid_c   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_c = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_c);
  GridCartesian*         FGrid_c   = SpaceTimeGrid::makeFiveDimGrid(nrhs, UGrid_c);
  GridRedBlackCartesian* FrbGrid_c = SpaceTimeGrid::makeFiveDimRedBlackGrid(nrhs, UGrid_c);
#else // 4d use case (f = fine, c = coarse)
  GridCartesian*         UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian*         FGrid_f   = UGrid_f;
  GridRedBlackCartesian* FrbGrid_f = UrbGrid_f;
  GridCartesian*         UGrid_c   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid_c = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_c);
  GridCartesian*         FGrid_c   = UGrid_c;
  GridRedBlackCartesian* FrbGrid_c = UrbGrid_c;
#endif

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

  typedef Grid::Upstream::Aggregation<vSpinColourVector, vTComplex, nBasis> UpstreamAggregation;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis> BaselineAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>               FourSpinAggregation;

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

  typedef TwoSpinAggregation TwoSpinAggregationMRHS; // just to differentiate profiler output

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  const int cb = 0;
  const int checkOrthog = 1;
  const int projectSpeed = 1;

  UpstreamAggregation UpstreamAggs(UGrid_c, UGrid_f, cb);
  TwoSpinAggregation TwoSpinAggsFast(UGrid_c, UGrid_f, cb, projectSpeed); // make it 4d for now (for benchmarking)

  UpstreamAggs.CreateSubspaceRandom(UPRNG_f);

  for(int i = 0; i < TwoSpinAggsFast.Subspace().size(); ++i)
    TwoSpinAggsFast.Subspace()[i] = UpstreamAggs.subspace[i];
  TwoSpinAggsFast.Orthogonalise(checkOrthog, 1); // 1 gs pass

  performChiralDoubling(UpstreamAggs.subspace);
  UpstreamAggs.Orthogonalise(checkOrthog, 1); // 1 gs pass

  /////////////////////////////////////////////////////////////////////////////
  //             Calculate numbers needed for performance figures            //
  /////////////////////////////////////////////////////////////////////////////

  auto USiteElems_f = getSiteElems<LatticeFermion>();
  auto USiteElems_c = getSiteElems<UpstreamCoarseVector>();

  std::cout << GridLogDebug << "USiteElems_f = " << USiteElems_f << std::endl;
  std::cout << GridLogDebug << "USiteElems_c = " << USiteElems_c << std::endl;

  double UVolume_f = std::accumulate(UGrid_f->_fdimensions.begin(), UGrid_f->_fdimensions.end(), 1, std::multiplies<double>());
  double UVolume_c = std::accumulate(UGrid_c->_fdimensions.begin(), UGrid_c->_fdimensions.end(), 1, std::multiplies<double>());

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for ProjectToSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    CoarseningLookupTable lut(UGrid_c, UGrid_f);

    std::vector<LatticeFermion> vecs_src_4d(nrhs, UGrid_f);
    std::vector<UpstreamCoarseVector> vecs_res_4d_improved(nrhs, UGrid_c);
    std::vector<TwoSpinCoarseVector> vecs_res_4d_twospin(nrhs, UGrid_c);

    LatticeFermion vecs_src_5d(FGrid_f);
    UpstreamCoarseVector vecs_res_5d_improved(FGrid_c);
    TwoSpinCoarseVector vecs_res_5d_twospin(FGrid_c);

    for(int i=0; i<nrhs; i++) {
      random(UPRNG_f, vecs_src_4d[i]);
      InsertSlice(vecs_src_4d[i], vecs_src_5d, i, 0);
      vecs_res_4d_improved[i] = Zero();
      vecs_res_4d_twospin[i] = Zero();
      InsertSlice(vecs_res_4d_improved[i], vecs_res_5d_improved, i, 0);
      InsertSlice(vecs_res_4d_twospin[i], vecs_res_5d_twospin, i, 0);
    }

    double flop = UVolume_f * (8 * USiteElems_f) * nBasis * nrhs;
    double byte = UVolume_f * (2 * 1 + 2 * USiteElems_f) * nBasis * sizeof(Complex) * nrhs;
    BenchmarkFunctionMRHS(Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct,
                          flop, byte, nIterMin, nSecMin, nrhs,
                          vecs_res_4d_improved[rhs], vecs_src_4d[rhs], UpstreamAggs.subspace, lut);

    BenchmarkFunction(Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct,
                      flop, byte, nIterMin, nSecMin,
                      vecs_res_5d_improved, vecs_src_5d, UpstreamAggs.subspace, lut);

    BenchmarkFunctionMRHS(TwoSpinAggregation::Kernels::aggregateProjectFast,
                          flop, byte, nIterMin, nSecMin, nrhs,
                          vecs_res_4d_twospin[rhs], vecs_src_4d[rhs], TwoSpinAggsFast.Subspace(), lut);

    BenchmarkFunction(TwoSpinAggregationMRHS::Kernels::aggregateProjectFast,
                      flop, byte, nIterMin, nSecMin,
                      vecs_res_5d_twospin, vecs_src_5d, TwoSpinAggsFast.Subspace(), lut);

    UpstreamCoarseVector tmp_improved(UGrid_c);
    for(int i=0; i<nrhs; i++) {
      ExtractSlice(tmp_improved, vecs_res_5d_improved, i, 0);
      printDeviationFromReference(tol, vecs_res_4d_improved[i], tmp_improved);
    }

    TwoSpinCoarseVector tmp_twospin(UGrid_c);
    for(int i=0; i<nrhs; i++) {
      ExtractSlice(tmp_twospin, vecs_res_5d_twospin, i, 0);
      printDeviationFromReference(tol, vecs_res_4d_twospin[i], tmp_twospin);
    }

    for(int i=0; i<nrhs; i++) {
      LayoutConverters::convertLayout(vecs_res_4d_twospin[i], tmp_improved);
      printDeviationFromReference(tol, vecs_res_4d_improved[i], tmp_improved);
    }
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for PromoteFromSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    CoarseningLookupTable lut(UGrid_c, UGrid_f);

    std::vector<TwoSpinCoarseVector> vecs_src_4d(nrhs, UGrid_c);
    std::vector<LatticeFermion> vecs_res_4d_twospin(nrhs, UGrid_f);

    TwoSpinCoarseVector vecs_src_5d(FGrid_c);
    LatticeFermion vecs_res_5d_twospin(FGrid_f);

    for(int i=0; i<nrhs; i++) {
      random(UPRNG_c, vecs_src_4d[i]);
      InsertSlice(vecs_src_4d[i], vecs_src_5d, i, 0);
      vecs_res_4d_twospin[i] = Zero();
      InsertSlice(vecs_res_4d_twospin[i], vecs_res_5d_twospin, i, 0);
    }

    double flop = UVolume_f * (8 * (nBasis - 1) + 6) * USiteElems_f * nrhs;
    double byte = UVolume_f * ((1 * 1 + 3 * USiteElems_f) * (nBasis - 1) + (1 * 1 + 2 * USiteElems_f) * 1) * sizeof(Complex) * nrhs;

    BenchmarkFunctionMRHS(TwoSpinAggregation::Kernels::aggregatePromoteFast,
                          flop, byte, nIterMin, nSecMin, nrhs,
                          vecs_src_4d[rhs], vecs_res_4d_twospin[rhs], TwoSpinAggsFast.Subspace(), lut);

    BenchmarkFunction(TwoSpinAggregationMRHS::Kernels::aggregatePromoteFast,
                      flop, byte, nIterMin, nSecMin,
                      vecs_src_5d, vecs_res_5d_twospin, TwoSpinAggsFast.Subspace(), lut);

    LatticeFermion tmp(UGrid_f);
    for(int i=0; i<nrhs; i++) {
      ExtractSlice(tmp, vecs_res_5d_twospin, i, 0);
      printDeviationFromReference(tol, vecs_res_4d_twospin[i], tmp);
    }
  }

  Grid_finalize();
}
