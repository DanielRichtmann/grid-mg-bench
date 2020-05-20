/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_blockMaskedInnerProduct.cc

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

using namespace Grid;
using namespace Grid::Rework;
using namespace Grid::BenchmarkHelpers;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_blockMaskedInnerProduct
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

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  const int cb = 0;
  const int checkOrthog = 1;
  const int projectSpeed = 1;

  UpstreamAggregation UpstreamAggs(UGrid_c, UGrid_f, cb);
  TwoSpinAggregation TwoSpinAggsFast(UGrid_c, UGrid_f, cb, projectSpeed);

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
  //                 Set up stuff required for the benchmark                 //
  /////////////////////////////////////////////////////////////////////////////

  typedef UpstreamCoarsenedMatrix::FineField::vector_object Fobj;
  typedef Lattice<iScalar<vInteger>> UpstreamCoor;
  typedef Lattice<typename Fobj::tensor_reduced> UpstreamFineComplexField;
  typedef typename Fobj::scalar_type UpstreamScalarType;
  typedef UpstreamCoarsenedMatrix::CoarseComplexField UpstreamCoarseComplexField;
  typedef UpstreamCoarsenedMatrix::CoarseVector UpstreamCoarseFermionField;

  typedef TwoSpinCoarsenedMatrix::FermionField TwoSpinCoarseFermionField;
  typedef TwoSpinCoarsenedMatrix::FineScalarField TwoSpinFineScalarField;

  Grid::Upstream::Geometry geom(UGrid_c->_ndimension);

  std::vector<UpstreamFineComplexField> masks(geom.npoint, UGrid_f);
  std::vector<CoarseningLookupTable> lut(geom.npoint);

  {
    UpstreamFineComplexField one(UGrid_f); one = UpstreamScalarType(1.0, 0.0);
    UpstreamFineComplexField zero(UGrid_f); zero = UpstreamScalarType(0.0, 0.0);
    UpstreamCoor             coor(UGrid_f);

    for(int p = 0; p < geom.npoint; ++p) {
      int     dir   = geom.directions[p];
      int     disp  = geom.displacements[p];
      Integer block = (UGrid_f->_rdimensions[dir] / UGrid_c->_rdimensions[dir]);

      LatticeCoordinate(coor, dir);

      if(disp == 0) {
        masks[p] = Zero();
      } else if(disp == 1) {
        masks[p] = where(mod(coor, block) == (block - 1), one, zero);
      } else if(disp == -1) {
        masks[p] = where(mod(coor, block) == (Integer)0, one, zero);
      }

      lut[p].populate(UGrid_c, masks[p]);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for ProjectToSubspace" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    LatticeFermion vec_src(UGrid_f);
    std::vector<UpstreamCoarseComplexField> vec_res_upstream(nBasis, UGrid_c);
    std::vector<TwoSpinCoarseFermionField> vec_res_twospin(TwoSpinAggregation::Ns_c, UGrid_c);
    std::vector<UpstreamCoarseFermionField> vec_res_improved(1, UGrid_c);

    random(UPRNG_f, vec_src);

    for(int myPoint=0; myPoint<geom.npoint; ++myPoint) {
      for(auto& elem : vec_res_upstream) elem = Zero();
      for(auto& elem : vec_res_twospin) elem = Zero();
      for(auto& elem : vec_res_improved) elem = Zero();

      auto workSites_f = real(TensorRemove(sum(masks[myPoint]))); // remove sites with zero in its mask from flop and byte counting
      std::cout << GridLogMessage << "point = " << myPoint
                << ": workSites, volume, ratio = "
                << workSites_f << ", " << UVolume_f << ", " << workSites_f / UVolume_f << std::endl;

      double flop = workSites_f * (8 * USiteElems_f) * nBasis;
      double byte = workSites_f * (2 * 1 + 2 * USiteElems_f) * nBasis * sizeof(Complex);

      BenchmarkFunctionMRHS(Grid::Upstream::blockMaskedInnerProduct,
                            flop, byte, nIterMin, nSecMin, nBasis,
                            vec_res_upstream[rhs], masks[myPoint], UpstreamAggs.subspace[rhs], vec_src);

      BenchmarkFunctionMRHS(Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct,
                            flop, byte, nIterMin, nSecMin, 1,
                            vec_res_improved[rhs], vec_src, UpstreamAggs.subspace, lut[myPoint]);

      BenchmarkFunctionMRHS(TwoSpinAggregation::Kernels::aggregateProjectFast,
                            flop, byte, nIterMin, nSecMin, TwoSpinCoarsenedMatrix::Ns_c,
                            vec_res_twospin[rhs], vec_src, TwoSpinAggsFast.Subspace(), lut[myPoint]);

      // TODO: Compare results
    }
  }

  Grid_finalize();
}
