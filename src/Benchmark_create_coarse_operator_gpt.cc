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
#include <CoarsenedMatrixUpstreamImprovedDirsaveForGPT.h>
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
  const int  nBlockL             = readFromCommandLineInt(&argc, &argv, "--nblockl", 2);
  const int  nBlockR             = readFromCommandLineInt(&argc, &argv, "--nblockr", 2);
  Coordinate blockSize           = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({4, 4, 4, 4}));
  bool       useClover           = readFromCommandLineToggle(&argc, &argv, "--clover");
  // clang-format on

  std::cout << GridLogMessage << "Using " << (useClover ? "clover" : "wilson") << " fermions" << std::endl;
  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;
  std::cout << GridLogMessage << "Using nBlockL = " << nBlockL << std::endl;
  std::cout << GridLogMessage << "Using nBlockR = " << nBlockR << std::endl;

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
  typedef Grid::UpstreamImprovedDirsaveForGPT::Aggregation<vSpinColourVector, vTComplex, nBasis>  ImprovedDirsaveForGPTAggregation;
  typedef Grid::UpstreamImprovedDirsaveLut::Aggregation<vSpinColourVector, vTComplex, nBasis>     ImprovedDirsaveLutAggregation;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::Aggregation<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutMRHSAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                                      OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                                      TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>                                     FourSpinAggregation;

  typedef Grid::Upstream::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                       UpstreamCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                       BaselineCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsave::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>        ImprovedDirsaveCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveForGPT::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>  ImprovedDirsaveForGPTCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLut::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>     ImprovedDirsaveLutCoarsenedMatrix;
  typedef Grid::UpstreamImprovedDirsaveLutMRHS::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> ImprovedDirsaveLutMRHSCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>                                      OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>                                      TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>                                     FourSpinCoarsenedMatrix;

  typedef UpstreamCoarsenedMatrix::CoarseVector               UpstreamCoarseVector;
  typedef BaselineCoarsenedMatrix::CoarseVector               BaselineCoarseVector;
  typedef ImprovedDirsaveCoarsenedMatrix::CoarseVector        ImprovedDirsaveCoarseVector;
  typedef ImprovedDirsaveForGPTCoarsenedMatrix::CoarseVector  ImprovedDirsaveForGPTCoarseVector;
  typedef ImprovedDirsaveLutCoarsenedMatrix::CoarseVector     ImprovedDirsaveLutCoarseVector;
  typedef ImprovedDirsaveLutMRHSCoarsenedMatrix::CoarseVector ImprovedDirsaveLutMRHSCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField                OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField                TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField               FourSpinCoarseVector;

  typedef UpstreamCoarsenedMatrix::CoarseMatrix               UpstreamCoarseLinkField;
  typedef BaselineCoarsenedMatrix::CoarseMatrix               BaselineCoarseLinkField;
  typedef ImprovedDirsaveCoarsenedMatrix::CoarseMatrix        ImprovedDirsaveCoarseLinkField;
  typedef ImprovedDirsaveForGPTCoarsenedMatrix::CoarseMatrix  ImprovedDirsaveForGPTCoarseLinkField;
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

  if(useClover) {
    LinOp = &LinOpDw;
  } else {
    LinOp  = &LinOpDwc;
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
  performChiralDoublingG5C(UpstreamAggs.subspace);
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
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for CoarsenOperator" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    uint64_t nIterOnce = 1;
    uint64_t nSecOnce  = 100;

    ImprovedDirsaveForGPTAggregation ImprovedDirsaveForGPTAggs(FGrid_c, FGrid_f, cb);
    for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) ImprovedDirsaveForGPTAggs.subspace[i] = UpstreamAggs.subspace[i];

    UpstreamCoarseLinkField CoarseLFUpstreamTmp(FGrid_c);

    double flop = 0; // TODO
    double byte = 0; // TODO

    assert(nBasis % nBlockL == 0);
    assert(nBasis % nBlockR == 0);

    int nVecPerBlockL = nBasis/nBlockL;
    int nVecPerBlockR = nBasis/nBlockR;

    // Upstream = Reference (always run) //////////////////////////////////////

    UpstreamCMat.CoarsenOperator(FGrid_f, *LinOp, UpstreamAggs);
    auto profResults = UpstreamCMat.GetProfile(); UpstreamCMat.ResetProfile();
    prettyPrintProfiling("Upstream", profResults, profResults["CoarsenOperator.Total"].t, false);

    // Improved version with full sets of vectors /////////////////////////////

    {
      ImprovedDirsaveForGPTCoarsenedMatrix ImprovedDirsaveForGPTCMat(*FGrid_c, isHermitian);

      ImprovedDirsaveForGPTCMat.CoarsenOperator(FGrid_f, *LinOp, ImprovedDirsaveForGPTAggs);
      profResults = ImprovedDirsaveForGPTCMat.GetProfile(); ImprovedDirsaveForGPTCMat.ResetProfile();
      prettyPrintProfiling("ImprovedDirsaveForGPT", profResults, profResults["CoarsenOperator.Total"].t, false);

      std::cout << GridLogMessage << "Deviations of ImprovedDirsaveForGPT from Upstream" << std::endl;
      for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
        assertResultMatchesReference(tol, UpstreamCMat.A[p], ImprovedDirsaveForGPTCMat.A[p]);
      }
    }

    // Improved version with only left vectors blocked ////////////////////////

    if(nBlockL > 1) {
      ImprovedDirsaveForGPTCoarsenedMatrix ImprovedDirsaveForGPT(*FGrid_c, isHermitian);

      std::vector<std::vector<LatticeFermion>> subspaceL(nBlockL, std::vector<LatticeFermion>(nVecPerBlockL, FGrid_f));
      for(int bL=0; bL<nBlockL; bL++) {
        for(int vL=0; vL<nVecPerBlockL; vL++) {
          subspaceL[bL][vL] = ImprovedDirsaveForGPTAggs.subspace[bL*nVecPerBlockL + vL];
        }
      }

      ImprovedDirsaveForGPT.ZeroLinks();
      for(int bL=0; bL<nBlockL; bL++) {
        ImprovedDirsaveForGPT.CoarsenOperator(FGrid_f, *LinOp, subspaceL[bL], ImprovedDirsaveForGPTAggs.subspace, bL*nVecPerBlockL, 0);
      }
      ImprovedDirsaveForGPT.ConstructRemainingLinks();
      profResults = ImprovedDirsaveForGPT.GetProfile(); ImprovedDirsaveForGPT.ResetProfile();
      prettyPrintProfiling("ImprovedDirsaveForGPTBlockL", profResults, profResults["CoarsenOperator.Total"].t, false);

      std::cout << GridLogMessage << "Deviations of ImprovedDirsaveForGPTBlockL from Upstream" << std::endl;
      for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
        assertResultMatchesReference(tol, UpstreamCMat.A[p], ImprovedDirsaveForGPT.A[p]);
      }
    }

    // Improved version with only right vectors blocked ///////////////////////

    if(nBlockR > 1) {
      ImprovedDirsaveForGPTCoarsenedMatrix ImprovedDirsaveForGPT(*FGrid_c, isHermitian);

      std::vector<std::vector<LatticeFermion>> subspaceR(nBlockR, std::vector<LatticeFermion>(nVecPerBlockR, FGrid_f));
      for(int bR=0; bR<nBlockR; bR++) {
        for(int vR=0; vR<nVecPerBlockR; vR++) {
          subspaceR[bR][vR] = ImprovedDirsaveForGPTAggs.subspace[bR*nVecPerBlockR + vR];
        }
      }

      ImprovedDirsaveForGPT.ZeroLinks();
      for(int bR=0; bR<nBlockR; bR++) {
        ImprovedDirsaveForGPT.CoarsenOperator(FGrid_f, *LinOp, ImprovedDirsaveForGPTAggs.subspace, subspaceR[bR], 0, bR*nVecPerBlockR);
      }
      ImprovedDirsaveForGPT.ConstructRemainingLinks();
      profResults = ImprovedDirsaveForGPT.GetProfile(); ImprovedDirsaveForGPT.ResetProfile();
      prettyPrintProfiling("ImprovedDirsaveForGPTBlockR", profResults, profResults["CoarsenOperator.Total"].t, false);

      std::cout << GridLogMessage << "Deviations of ImprovedDirsaveForGPTBlockR from Upstream" << std::endl;
      for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
        assertResultMatchesReference(tol, UpstreamCMat.A[p], ImprovedDirsaveForGPT.A[p]);
      }
    }

    // Improved version with both left and right vectors blocked //////////////

    if(nBlockL > 1 && nBlockR > 1) {
      ImprovedDirsaveForGPTCoarsenedMatrix ImprovedDirsaveForGPT(*FGrid_c, isHermitian);

      std::vector<std::vector<LatticeFermion>> subspaceL(nBlockL, std::vector<LatticeFermion>(nVecPerBlockL, FGrid_f));
      for(int bL=0; bL<nBlockL; bL++) {
        for(int vL=0; vL<nVecPerBlockL; vL++) {
          subspaceL[bL][vL] = ImprovedDirsaveForGPTAggs.subspace[bL*nVecPerBlockL + vL];
        }
      }

      std::vector<std::vector<LatticeFermion>> subspaceR(nBlockR, std::vector<LatticeFermion>(nVecPerBlockR, FGrid_f));
      for(int bR=0; bR<nBlockR; bR++) {
        for(int vR=0; vR<nVecPerBlockR; vR++) {
          subspaceR[bR][vR] = ImprovedDirsaveForGPTAggs.subspace[bR*nVecPerBlockR + vR];
        }
      }

      ImprovedDirsaveForGPT.ZeroLinks();
      for(int bL=0; bL<nBlockL; bL++) {
        for(int bR=0; bR<nBlockR; bR++) {
          ImprovedDirsaveForGPT.CoarsenOperator(FGrid_f, *LinOp, subspaceL[bL], subspaceR[bR], bL*nVecPerBlockL, bR*nVecPerBlockR);
        }
      }
      ImprovedDirsaveForGPT.ConstructRemainingLinks();
      profResults = ImprovedDirsaveForGPT.GetProfile(); ImprovedDirsaveForGPT.ResetProfile();
      prettyPrintProfiling("ImprovedDirsaveForGPTBlockLR", profResults, profResults["CoarsenOperator.Total"].t, false);

      std::cout << GridLogMessage << "Deviations of ImprovedDirsaveForGPTBlockLR from Upstream" << std::endl;
      for(int p = 0; p < UpstreamCMat.geom.npoint; ++p) {
        assertResultMatchesReference(tol, UpstreamCMat.A[p], ImprovedDirsaveForGPT.A[p]);
      }
    }

  }

  Grid_finalize();
}
