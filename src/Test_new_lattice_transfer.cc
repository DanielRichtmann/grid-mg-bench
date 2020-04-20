/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/Test_new_lattice_transfer.cc

    Copyright (C) 2015 - 2019

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

using namespace Grid;
using namespace Grid::Rework;

// TODO: temporary, remove later!
Coordinate calcCoarseLattSize(const Coordinate& fineLattSize, const Coordinate& blockSize) {
  Coordinate ret(fineLattSize);
  for(int d = 0; d < ret.size(); d++) { ret[d] /= blockSize[d]; }
  return ret;
}

template<class CoarseningPolicy>
void printCoarseningPolicy() {
  std::cout << "CoarseningPolicy::Ns_f = " << CoarseningPolicy::Ns_f << std::endl;
  std::cout << "CoarseningPolicy::Ns_c = " << CoarseningPolicy::Ns_c << std::endl;
  std::cout << "CoarseningPolicy::Ns_b = " << CoarseningPolicy::Ns_b << std::endl;
  std::cout << "CoarseningPolicy::Nc_f = " << CoarseningPolicy::Nc_f << std::endl;
  std::cout << "CoarseningPolicy::Nc_c = " << CoarseningPolicy::Nc_c << std::endl;
}

#define DO_TYPEDEFS(Nbasis) \
  typedef CoarseningPolicy<LatticeFermion, Nbasis, 1> OneSpinCPolicy; \
  typedef CoarseningPolicy<LatticeFermion, Nbasis, 2> TwoSpinCPolicy; \
  typedef CoarseningPolicy<LatticeFermion, Nbasis, 4> FourSpinCPolicy; \
\
  typedef Grid::Rework::Aggregation<OneSpinCPolicy>  OneSpinAggregation; \
  typedef Grid::Rework::Aggregation<TwoSpinCPolicy>  TwoSpinAggregation; \
  typedef Grid::Rework::Aggregation<FourSpinCPolicy> FourSpinAggregation; \
\
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCPolicy>  OneSpinCoarsenedMatrix;  \
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCPolicy>  TwoSpinCoarsenedMatrix;  \
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCPolicy> FourSpinCoarsenedMatrix; \
\
  typedef typename OneSpinAggregation::FermionField  OneSpinCoarseFermionField; \
  typedef typename TwoSpinAggregation::FermionField  TwoSpinCoarseFermionField; \
  typedef typename FourSpinAggregation::FermionField FourSpinCoarseFermionField;

#define DO_AGGREGATION_SETUP_AND_TESTS() \
  do { \
    CoarseningAggregation aggregation(CGrid, FGrid, 0); \
\
    aggregation.Create(basisVectors);      \
\
    CoarseFermionField psi_c(CGrid); \
    psi_c = Zero(); \
    CoarseFermionField src_c(CGrid); \
    src_c = Zero(); \
    CoarseFermionField diff_c(CGrid); \
    diff_c = Zero(); \
\
    auto tolerance = 1e-15; \
\
    { /* first test */ \
      for(int i = 0; i < basisVectors().size(); ++i) { \
        aggregation.ProjectToSubspace(psi_c, aggregation.Subspace()[i]); \
        aggregation.PromoteFromSubspace(psi_c, psi_f); \
\
        diff_f         = aggregation.Subspace()[i] - psi_f; \
        auto deviation = std::sqrt(norm2(diff_f) / norm2(aggregation.Subspace()[i])); \
        std::cout << "Vector " << i << ": norm2(v_i) = " << norm2(aggregation.Subspace()[i]) \
                  << " | norm2(R v_i) = " << norm2(psi_c) << " | norm2(P R v_i) = " << norm2(psi_f) \
                  << " | relative deviation = " << deviation; \
\
        if(deviation > tolerance) { \
          std::cout << " > " << tolerance << " -> check failed" << std::endl; \
          abort(); \
        } else { \
          std::cout << " < " << tolerance << " -> check passed" << std::endl; \
        } \
      } \
    } \
\
    { /* second test */ \
      random(CpRNG, src_c); \
\
      aggregation.PromoteFromSubspace(src_c, psi_f); \
      aggregation.ProjectToSubspace(psi_c, psi_f); \
\
      diff_c         = src_c - psi_c; \
      auto deviation = std::sqrt(norm2(diff_c) / norm2(src_c)); \
\
      std::cout << "norm2(v_c) = " << norm2(src_c) << " | norm2(R P v_c) = " << norm2(psi_c) \
                << " | norm2(P v_c) = " << norm2(psi_f) << " | relative deviation = " << deviation; \
\
      if(deviation > tolerance) { \
        std::cout << " > " << tolerance << " -> check failed" << std::endl; \
        abort(); \
      } else { \
        std::cout << " < " << tolerance << " -> check passed" << std::endl; \
      } \
    } \
  } while(0)

#define DO_COARSENEDMATRIX_SETUP_AND_TESTS() \
  do { \
    CoarseningAggregation aggregation(CGrid, FGrid, 0); \
    aggregation.Create(basisVectors); \
\
    CoarseningCoarsenedMatrix coarsenedMatrix(*CGrid, *CRBGrid); \
    coarsenedMatrix.CoarsenOperator(FGrid, LinOp, basisVectors, aggregation); \
\
    MdagMLinearOperator<CoarseningCoarsenedMatrix, CoarseFermionField> LinOp(coarsenedMatrix); \
  } while(0)

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const Coordinate blockSize  = std::vector<int>({2, 2, 2, 2});
  const Coordinate lattsize_f = GridDefaultLatt();
  const Coordinate lattsize_c = calcCoarseLattSize(lattsize_f, blockSize);

  GridCartesian* FGrid = SpaceTimeGrid::makeFourDimGrid(
    lattsize_f, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian* CGrid =
    SpaceTimeGrid::makeFourDimGrid(lattsize_c, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* FRBGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);
  GridRedBlackCartesian* CRBGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG FpRNG(FGrid);
  GridParallelRNG CpRNG(CGrid);
  FpRNG.SeedFixedIntegers(seeds);
  CpRNG.SeedFixedIntegers(seeds);

  // clang-format off
  LatticeFermion    src_f(FGrid); random(FpRNG, src_f);
  LatticeFermion    psi_f(FGrid); psi_f = Zero();
  LatticeFermion    diff_f(FGrid); diff_f = Zero();
  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FpRNG, Umu);
  // clang-format on

  typedef typename WilsonCloverFermionR::FermionField FermionField;
  typename WilsonCloverFermionR::ImplParams params;
  WilsonAnisotropyCoefficients anis;

  RealD                mass  = 0.5;
  RealD                csw_r = 1.0;
  RealD                csw_t = 1.0;
  WilsonCloverFermionR Dwc(Umu, *FGrid, *FRBGrid, mass, csw_r, csw_t, anis, params);

  MdagMLinearOperator<WilsonCloverFermionR, LatticeFermion> LinOp(Dwc);
  TrivialPrecon<LatticeFermion>                             simple;
  FlexibleGeneralisedMinimalResidual<LatticeFermion>        FGMRES(1.0e-14, 4, simple, 4, false);

  const int Nbasis = 16;

  MGBasisVectorsParams bvPar;
  bvPar.preOrthonormalise = false;
  bvPar.postOrthonormalise = true;
  bvPar.testVectorSetup = true;
  bvPar.maxIter = 1;

  MGBasisVectors<LatticeFermion> basisVectors(FGrid, 0, Nbasis);
  basisVectors.InitRandom(FpRNG);
  basisVectors.Generate(FGMRES, LinOp, bvPar, [](int n){}, [](){}); // in initial setup we don't need any modifications via the callbacks

  DO_TYPEDEFS(Nbasis);

  {
    typedef OneSpinCoarseFermionField CoarseFermionField;
    typedef OneSpinAggregation        CoarseningAggregation;
    typedef OneSpinCoarsenedMatrix    CoarseningCoarsenedMatrix;
    printCoarseningPolicy<OneSpinCPolicy>();
    DO_AGGREGATION_SETUP_AND_TESTS();
    DO_COARSENEDMATRIX_SETUP_AND_TESTS();
  }
  {
    typedef TwoSpinCoarseFermionField CoarseFermionField;
    typedef TwoSpinAggregation        CoarseningAggregation;
    typedef TwoSpinCoarsenedMatrix    CoarseningCoarsenedMatrix;
    printCoarseningPolicy<TwoSpinCPolicy>();
    DO_AGGREGATION_SETUP_AND_TESTS();
    DO_COARSENEDMATRIX_SETUP_AND_TESTS();
  }
  {
    typedef FourSpinCoarseFermionField CoarseFermionField;
    typedef FourSpinAggregation        CoarseningAggregation;
    typedef FourSpinCoarsenedMatrix    CoarseningCoarsenedMatrix;
    printCoarseningPolicy<FourSpinCPolicy>();
    DO_AGGREGATION_SETUP_AND_TESTS();
    DO_COARSENEDMATRIX_SETUP_AND_TESTS();
  }

  // It seems we have set everything up correctly -> TODO now work on the CoarsenedMatrix rework

  Grid_finalize();
}
