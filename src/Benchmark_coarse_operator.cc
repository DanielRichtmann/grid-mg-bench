/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_coarse_operator.cc

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
#include <CoarsenedMatrixUpstreamImproved.h>
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
  const int  nBasis = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int  nB     = nBasis / 2;
  int        nIter  = readFromCommandLineInt(&argc, &argv, "--niter", 1000);
  // clang-format on

  std::cout << GridLogMessage << "Compiled with nBasis = " << nBasis << " -> nB = " << nB << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  GridCartesian*         CGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  CGrid->show_decomposition();

  GridParallelRNG CPRNG(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  CPRNG.SeedFixedIntegers(seeds);

  RealD tol = getPrecision<vComplex>::value == 2 ? 1e-15 : 1e-7;

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningPolicy<LatticeFermion, nB, 1> OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2> TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4> FourSpinCoarseningPolicy;

  typedef Grid::Upstream::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>         UpstreamCoarsenedMatrix;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>         BaselineCoarsenedMatrix;
  typedef Grid::UpstreamImproved::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> ImprovedCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<OneSpinCoarseningPolicy>                        OneSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy>                        TwoSpinCoarsenedMatrix;
  typedef Grid::Rework::CoarsenedMatrix<FourSpinCoarseningPolicy>                       FourSpinCoarsenedMatrix;

  typedef UpstreamCoarsenedMatrix::CoarseVector UpstreamCoarseVector;
  typedef BaselineCoarsenedMatrix::CoarseVector BaselineCoarseVector;
  typedef ImprovedCoarsenedMatrix::CoarseVector ImprovedCoarseVector;
  typedef OneSpinCoarsenedMatrix::FermionField  OneSpinCoarseVector;
  typedef TwoSpinCoarsenedMatrix::FermionField  TwoSpinCoarseVector;
  typedef FourSpinCoarsenedMatrix::FermionField FourSpinCoarseVector;

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  const int hermitian = 0;

  UpstreamCoarsenedMatrix UpstreamCMat(*CGrid, hermitian);
  // BaselineCoarsenedMatrix BaselineCMat(*CGrid, *CrbGrid, hermitian);
  // ImprovedCoarsenedMatrix ImprovedCMat(*CGrid, hermitian);
  TwoSpinCoarsenedMatrix  TwoSpinCMat(*CGrid, *CrbGrid, 2, hermitian); // speedLevel = 2 (affects only CoarsenOperator)

  for(int p = 0; p < TwoSpinCMat.geom_.npoint; ++p) {
    random(CPRNG, TwoSpinCMat.Y_[p]);
    convertLayout(TwoSpinCMat.Y_[p], UpstreamCMat.A[p]);
    // BaselineCMat.A[p] = UpstreamCMat.A[p];
    // ImprovedCMat.A[p] = UpstreamCMat.A[p];
  }

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil   = UpstreamCMat.geom.npoint;
  double nAccum     = nStencil - 1;
  double CSiteElems = getSiteElems<UpstreamCoarseVector>();

  std::cout << GridLogDebug << "CSiteElems = " << CSiteElems << std::endl;

  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for M" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    UpstreamCoarseVector CoarseVecUpstreamIn(CGrid);  random(CPRNG, CoarseVecUpstreamIn);
    TwoSpinCoarseVector  CoarseVecTwospinIn(CGrid);   convertLayout(CoarseVecUpstreamIn, CoarseVecTwospinIn);
    UpstreamCoarseVector CoarseVecUpstreamOut(CGrid); CoarseVecUpstreamOut = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinOut(CGrid);  CoarseVecTwospinOut  = Zero();
    UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
    // clang-format on

    // int osites=Grid()->oSites();
    // double flops = osites*Nsimd*nbasis*nbasis*8.0*geom.npoint;
    // double bytes = osites*nbasis*nbasis*geom.npoint*sizeof(CComplex);

    int osites = CGrid->oSites();
    double flop_peter = osites*vTComplex::Nsimd()*nBasis*nBasis*8.0*UpstreamCMat.geom.npoint;
    double byte_peter = osites*nBasis*nBasis*UpstreamCMat.geom.npoint*sizeof(vTComplex);

    double flop = CVolume * ((nStencil * (8 * CSiteElems * CSiteElems - 2 * CSiteElems) + nAccum * 2 * CSiteElems) + 8 * CSiteElems);
    double byte = CVolume * ((nStencil * (CSiteElems * CSiteElems + CSiteElems) + CSiteElems) + CSiteElems) * sizeof(Complex);

    std::cout << GridLogMessage << "mine flop, byte: " << flop << ", " << byte << std::endl;
    std::cout << GridLogMessage << "peter flop, byte: " << flop_peter << ", " << byte_peter << std::endl;

    BenchmarkFunction(UpstreamCMat.M, flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMat.M,  flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

    convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
    printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Mdag" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    UpstreamCoarseVector CoarseVecUpstreamIn(CGrid);  random(CPRNG, CoarseVecUpstreamIn);
    TwoSpinCoarseVector  CoarseVecTwospinIn(CGrid);   convertLayout(CoarseVecUpstreamIn, CoarseVecTwospinIn);
    UpstreamCoarseVector CoarseVecUpstreamOut(CGrid); CoarseVecUpstreamOut = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinOut(CGrid);  CoarseVecTwospinOut  = Zero();
    UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
    // clang-format on

    // NOTE: these values are based on Galerkin coarsening, i.e., Mdag = g5c * M * g5c
    double flop = CVolume * ((nStencil * (8 * CSiteElems * CSiteElems - 2 * CSiteElems) + nAccum * 2 * CSiteElems) + 8 * CSiteElems) + 2 * CVolume * (3 * CSiteElems);
    double byte = CVolume * ((nStencil * (CSiteElems * CSiteElems + CSiteElems) + CSiteElems) + CSiteElems) * sizeof(Complex) + 2 * CVolume * (3 * CSiteElems) * sizeof(Complex);

    BenchmarkFunction(UpstreamCMat.Mdag, flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMat.Mdag,  flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

    convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
    printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Mdir" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    UpstreamCoarseVector CoarseVecUpstreamIn(CGrid);  random(CPRNG, CoarseVecUpstreamIn);
    TwoSpinCoarseVector  CoarseVecTwospinIn(CGrid);   convertLayout(CoarseVecUpstreamIn, CoarseVecTwospinIn);
    UpstreamCoarseVector CoarseVecUpstreamOut(CGrid); CoarseVecUpstreamOut = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinOut(CGrid);  CoarseVecTwospinOut  = Zero();
    UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
    // clang-format on

    double flop = CVolume * (8 * CSiteElems * CSiteElems - 2 * CSiteElems);
    double byte = CVolume * (CSiteElems * CSiteElems + 2 * CSiteElems) * sizeof(Complex);

    BenchmarkFunction(UpstreamCMat.Mdir, flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut, 2, 1);
    BenchmarkFunction(TwoSpinCMat.Mdir,  flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut,  2, 1);

    convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
    printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for Mdiag" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    UpstreamCoarseVector CoarseVecUpstreamIn(CGrid);  random(CPRNG, CoarseVecUpstreamIn);
    TwoSpinCoarseVector  CoarseVecTwospinIn(CGrid);   convertLayout(CoarseVecUpstreamIn, CoarseVecTwospinIn);
    UpstreamCoarseVector CoarseVecUpstreamOut(CGrid); CoarseVecUpstreamOut = Zero();
    TwoSpinCoarseVector  CoarseVecTwospinOut(CGrid);  CoarseVecTwospinOut  = Zero();
    UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
    // clang-format on

    double flop = CVolume * (8 * CSiteElems * CSiteElems - 2 * CSiteElems);
    double byte = CVolume * (CSiteElems * CSiteElems + 2 * CSiteElems) * sizeof(Complex);

    BenchmarkFunction(UpstreamCMat.Mdiag, flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMat.Mdiag,  flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

    convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
    printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  }

  // {
  //   std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  //   std::cout << GridLogMessage << "Running benchmark for MdirAll" << std::endl;
  //   std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  //   // clang-format off
  //   <UpstreamCoarseVector> CoarseVecUpstreamIn(CGrid);  random(CPRNG, CoarseVecUpstreamIn);
  //   TwoSpinCoarseVector  CoarseVecTwospinIn(CGrid);   convertLayout(CoarseVecUpstreamIn, CoarseVecTwospinIn);
  //   std::vector<UpstreamCoarseVector> CoarseVecUpstreamOut(nStencil-1, CGrid); for(auto& elem : CoarseVecUpstreamOut) elem = Zero();
  //   std::vector<TwoSpinCoarseVector>  CoarseVecTwospinOut(nStencil-1, CGrid);  for(auto& elem : CoarseVecTwoSpinOut) elem = Zero();
  //   UpstreamCoarseVector CoarseVecUpstreamTmp(CGrid);
  //   // clang-format on

  //   double flop = CVolume * (8 * CSiteElems * CSiteElems - 2 * CSiteElems);
  //   double byte = CVolume * (CSiteElems * CSiteElems + 2 * CSiteElems) * sizeof(Complex);

  //   BenchmarkFunction(UpstreamCMat.Mdir, flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut, 2, 1);
  //   BenchmarkFunction(TwoSpinCMat.Mdir,  flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut,  2, 1);

  //   convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
  //   printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  // }

  Grid_finalize();
}
