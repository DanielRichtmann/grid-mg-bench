/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_blocking_operators.cc

    Copyright (C) 2015 - 2020

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
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_blocking_operators
#ifndef NBASIS
#define NBASIS 40
#endif

template<class CComplex, int nbasis>
void convertLayout(std::vector<Lattice<CComplex>> const& in, Lattice<iVector<CComplex, nbasis>>& out) {
  assert(in.size() == nbasis);
  for(auto const& elem : in) conformable(elem, out);

  auto  out_v = out.View();
  auto  in_vc = getViewContainer(in);
  auto* in_va = &in_vc[0];
  accelerator_for(ss, out.Grid()->oSites(), CComplex::Nsimd(), {
    for(int i = 0; i < nbasis; i++) { coalescedWrite(out_v[ss](i), in_va[i](ss)); }
  });
}

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
  int        Ls        = readFromCommandLineInt(&argc, &argv, "--Ls", 1);
  uint64_t   nIterMin  = readFromCommandLineInt(&argc, &argv, "--miniter", 1000);
  uint64_t   nSecMin   = readFromCommandLineInt(&argc, &argv, "--minsec", 5);
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

  UpstreamAggregation UpstreamAggs(FGrid_c, FGrid_f, cb);
  TwoSpinAggregation TwoSpinAggsFast(FGrid_c, FGrid_f, cb, projectSpeed);

  UpstreamAggs.CreateSubspaceRandom(FPRNG_f);

  for(int i = 0; i < TwoSpinAggsFast.Subspace().size(); ++i)
    TwoSpinAggsFast.Subspace()[i] = UpstreamAggs.subspace[i];
  TwoSpinAggsFast.Orthogonalise(checkOrthog, 1); // 1 gs pass

  if(Ls != 1) performChiralDoublingG5R5(UpstreamAggs.subspace);
  else        performChiralDoublingG5C(UpstreamAggs.subspace);
  UpstreamAggs.Orthogonalise(checkOrthog, 1); // 1 gs pass

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

  Grid::Upstream::Geometry geom(FGrid_c->_ndimension);

  std::vector<UpstreamFineComplexField> masks(geom.npoint, FGrid_f);
  std::vector<CoarseningLookupTable> lut(geom.npoint);

  { // original setup code taken from CoarsenedMatrix
    UpstreamFineComplexField one(FGrid_f); one = UpstreamScalarType(1.0, 0.0);
    UpstreamFineComplexField zero(FGrid_f); zero = UpstreamScalarType(0.0, 0.0);
    UpstreamCoor             coor(FGrid_f);

    for(int p=0; p<geom.npoint; ++p) {
      int     dir   = geom.directions[p];
      int     disp  = geom.displacements[p];
      Integer block = (FGrid_f->_rdimensions[dir] / FGrid_c->_rdimensions[dir]);

      LatticeCoordinate(coor, dir);

      if(disp == 0) {
        masks[p] = Zero();
      } else if(disp == 1) {
        masks[p] = where(mod(coor,block) == (block-1), one, zero);
      } else if(disp == -1) {
        masks[p] = where(mod(coor,block) == (Integer)0, one, zero);
      }

      lut[p].populate(FGrid_c, masks[p]);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for blockMaskedInnerProduct" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    LatticeFermion src(FGrid_f);
    std::vector<UpstreamCoarseComplexField> res_upstream(nBasis, FGrid_c);
    std::vector<TwoSpinCoarseFermionField>  res_twospin(TwoSpinAggregation::Ns_c, FGrid_c);
    std::vector<UpstreamCoarseFermionField> res_improved(1, FGrid_c);

    random(FPRNG_f, src);

    for(int myPoint=0; myPoint<geom.npoint; ++myPoint) {
      for(auto& elem : res_upstream) elem = Zero();
      for(auto& elem : res_twospin) elem = Zero();
      for(auto& elem : res_improved) elem = Zero();

      auto workSites_f = real(TensorRemove(sum(masks[myPoint]))); // remove sites with zero in its mask from flop and byte counting

      std::cout << GridLogMessage << "point = " << myPoint
                << ": workSites, volume, ratio = "
                << workSites_f << ", " << FVolume_f << ", " << workSites_f / FVolume_f << std::endl;

      double flop = workSites_f * (8 * siteElems_f) * nBasis;
      double byte = workSites_f * (2 * 1 + 2 * siteElems_f) * nBasis * sizeof(Complex);

      BenchmarkFunctionMRHS(Grid::Upstream::blockMaskedInnerProduct,
                            flop, byte, nIterMin, nSecMin, nBasis,
                            res_upstream[rhs], masks[myPoint], UpstreamAggs.subspace[rhs], src);

      BenchmarkFunctionMRHS(Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct,
                            flop, byte, nIterMin, nSecMin, 1,
                            res_improved[rhs], src, UpstreamAggs.subspace, lut[myPoint]);

      BenchmarkFunctionMRHS(TwoSpinAggregation::Kernels::aggregateProjectFast,
                            flop, byte, nIterMin, nSecMin, TwoSpinCoarsenedMatrix::Ns_c,
                            res_twospin[rhs], src, TwoSpinAggsFast.Subspace(), lut[myPoint]);

      if(myPoint != geom.npoint-1) { // does nothing for self stencil point
        UpstreamCoarseFermionField tmp(FGrid_c);
        convertLayout(res_upstream, tmp);
        assertResultMatchesReference(tol, res_improved[0], tmp);
      }
    }
    std::cout << GridLogMessage << "Results are equal" << std::endl;
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for blockProject" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    LatticeFermion src(FGrid_f);
    std::vector<UpstreamCoarseFermionField> res_upstream_blockProject(1, FGrid_c);
    std::vector<UpstreamCoarseComplexField> res_upstream_blockMaskedInnerProduct(nBasis, FGrid_c);
    std::vector<TwoSpinCoarseFermionField>  res_twospin(TwoSpinAggregation::Ns_c, FGrid_c);
    std::vector<UpstreamCoarseFermionField> res_improved(1, FGrid_c);

    random(FPRNG_f, src);

    for(auto& elem : res_upstream_blockProject) elem = Zero();
    for(auto& elem : res_upstream_blockMaskedInnerProduct) elem = Zero();
    for(auto& elem : res_twospin) elem = Zero();
    for(auto& elem : res_improved) elem = Zero();

    UpstreamFineComplexField fullMask(FGrid_f); fullMask = UpstreamScalarType(1.0, 0.0);
    CoarseningLookupTable    fullLut(FGrid_c, fullMask);

    double flop = FVolume_f * (8 * siteElems_f) * nBasis;
    double byte = FVolume_f * (2 * 1 + 2 * siteElems_f) * nBasis * sizeof(Complex);

    BenchmarkFunctionMRHS(Grid::blockProject,
                          flop, byte, nIterMin, nSecMin, 1,
                          res_upstream_blockProject[rhs], src, UpstreamAggs.subspace);

    BenchmarkFunctionMRHS(Grid::Upstream::blockMaskedInnerProduct,
                          flop, byte, nIterMin, nSecMin, nBasis,
                          res_upstream_blockMaskedInnerProduct[rhs], fullMask, UpstreamAggs.subspace[rhs], src);

    BenchmarkFunctionMRHS(Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct,
                          flop, byte, nIterMin, nSecMin, 1,
                          res_improved[rhs], src, UpstreamAggs.subspace, fullLut);

    BenchmarkFunctionMRHS(TwoSpinAggregation::Kernels::aggregateProjectFast,
                          flop, byte, nIterMin, nSecMin, TwoSpinCoarsenedMatrix::Ns_c,
                          res_twospin[rhs], src, TwoSpinAggsFast.Subspace(), fullLut);

    UpstreamCoarseFermionField tmp(FGrid_c);
    convertLayout(res_upstream_blockMaskedInnerProduct, tmp);
    assertResultMatchesReference(tol, res_upstream_blockProject[0], tmp);
    assertResultMatchesReference(tol, res_upstream_blockProject[0], res_improved[0]);

    std::cout << GridLogMessage << "Results are equal" << std::endl;
  }

  Grid_finalize();
}
