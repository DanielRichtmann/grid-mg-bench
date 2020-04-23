/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_coarsenedmatrix.cc

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
#else // 4d use case (f = fine, c = coarse), fermion same as gauge
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
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

#if defined(FIVE_DIMENSIONS)
  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(UPRNG, Umu);

  RealD mass = 0.001;
  RealD M5   = 1.8;

  DomainWallFermionR Ddwf(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5);
  MdagMLinearOperator<DomainWallFermionR,LatticeFermion> LinOp(Ddwf);
#else
  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG, Umu);

  RealD mass = 0.5;

  WilsonFermionR                                      Dw(Umu, *FGrid, *FrbGrid, mass);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> LinOp(Dw);
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningPolicy<LatticeFermion, nB, 1> OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2> TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4> FourSpinCoarseningPolicy;

  typedef Grid::Upstream::Aggregation<vSpinColourVector, vTComplex, nBasis>         UpstreamAggregation;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis>         BaselineAggregation;
  typedef Grid::UpstreamImproved::Aggregation<vSpinColourVector, vTComplex, nBasis> ImprovedAggregation;
  typedef Grid::Rework::Aggregation<OneSpinCoarseningPolicy>                        OneSpinAggregation;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>                        TwoSpinAggregation;
  typedef Grid::Rework::Aggregation<FourSpinCoarseningPolicy>                       FourSpinAggregation;

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

  typedef UpstreamCoarsenedMatrix::CoarseMatrix UpstreamCoarseLinkField;
  typedef BaselineCoarsenedMatrix::CoarseMatrix BaselineCoarseLinkField;
  typedef ImprovedCoarsenedMatrix::CoarseMatrix ImprovedCoarseLinkField;
  typedef OneSpinCoarsenedMatrix::LinkField     OneSpinCoarseLinkField;
  typedef TwoSpinCoarsenedMatrix::LinkField     TwoSpinCoarseLinkField;
  typedef FourSpinCoarsenedMatrix::LinkField    FourSpinCoarseLinkField;

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamAggregation UpstreamAggs(CGrid, FGrid, 0);
  BaselineAggregation BaselineAggs(CGrid, FGrid, 0);
  ImprovedAggregation ImprovedAggs(CGrid, FGrid, 0);
  TwoSpinAggregation  TwoSpinAggsDefault(CGrid, FGrid, 0, 0);
  TwoSpinAggregation  TwoSpinAggsFast(CGrid, FGrid, 0, 1);

  UpstreamAggs.CreateSubspaceRandom(FPRNG);
  for(int i = 0; i < TwoSpinAggsDefault.Subspace().size(); ++i) {
    TwoSpinAggsDefault.Subspace()[i] = UpstreamAggs.subspace[i];
    TwoSpinAggsFast.Subspace()[i]    = UpstreamAggs.subspace[i];
  }
  for(int i = 0; i < UpstreamAggs.subspace.size(); ++i) {
    BaselineAggs.subspace[i] = UpstreamAggs.subspace[i];
    ImprovedAggs.subspace[i] = UpstreamAggs.subspace[i];
  }
  performChiralDoubling(UpstreamAggs.subspace);
  performChiralDoubling(BaselineAggs.subspace);
  performChiralDoubling(ImprovedAggs.subspace);

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  UpstreamCoarsenedMatrix UpstreamCMat(*CGrid, 0);           // hermitian = 0
  BaselineCoarsenedMatrix BaselineCMat(*CGrid, *CrbGrid, 0); // hermitian = 0
  ImprovedCoarsenedMatrix ImprovedCMat(*CGrid, 0);           // hermitian = 0
  TwoSpinCoarsenedMatrix  TwoSpinCMatSpeedLevel0(*CGrid, *CrbGrid, 0, 0); // speedLevel = 0, hermitian = 0
  TwoSpinCoarsenedMatrix  TwoSpinCMatSpeedLevel1(*CGrid, *CrbGrid, 1, 0); // speedLevel = 1, hermitian = 0
  TwoSpinCoarsenedMatrix  TwoSpinCMatSpeedLevel2(*CGrid, *CrbGrid, 2, 0); // speedLevel = 2, hermitian = 0

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil   = UpstreamCMat.geom.npoint;
  double nAccum     = nStencil;
  double FSiteElems = Nc * Ns;
  double CSiteElems = nBasis;

  double FVolume = std::accumulate(FGrid->_fdimensions.begin(), FGrid->_fdimensions.end(), 1, std::multiplies<double>());
  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

#if 0 // TODO DELETE AGAIN
  for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
    random(CPRNG, BaselineCMat.A[p]);
    convertLayout(BaselineCMat.A[p], TwoSpinCMatSpeedLevel0.Y_[p]);
  }
#else
  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Running benchmark for CoarsenOperator" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    auto nIterOne = 1;

    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel0DefaultProjects = TwoSpinCMatSpeedLevel0;
    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel0FastProjects    = TwoSpinCMatSpeedLevel0;
    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel1DefaultProjects = TwoSpinCMatSpeedLevel1;
    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel1FastProjects    = TwoSpinCMatSpeedLevel1;
    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel2DefaultProjects = TwoSpinCMatSpeedLevel2;
    TwoSpinCoarsenedMatrix& TwoSpinCMatSpeedLevel2FastProjects    = TwoSpinCMatSpeedLevel2;
    BaselineCoarseLinkField CoarseLFBaselineTmp(CGrid);

    double flop = 0; // TODO
    double byte = 0; // TODO

    BenchmarkFunction(BaselineCMat.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, BaselineAggs);
    auto profResults = BaselineCMat.GetProfile(); BaselineCMat.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);

    // NOTE: For some reason, this binary crashes on GPU with a bus error when I uncomment this
    BenchmarkFunction(UpstreamCMat.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, UpstreamAggs);
    profResults = UpstreamCMat.GetProfile(); UpstreamCMat.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    std::cout << GridLogMessage << "Deviations of upstream from baseline" << std::endl;
    for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
      printDeviationFromReference(tol, BaselineCMat.A[p], UpstreamCMat.A[p]);
    }

    BenchmarkFunction(ImprovedCMat.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, ImprovedAggs);
    profResults = ImprovedCMat.GetProfile(); ImprovedCMat.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    std::cout << GridLogMessage << "Deviations of improvedupstream from baseline" << std::endl;
    for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
      printDeviationFromReference(tol, BaselineCMat.A[p], ImprovedCMat.A[p]);
    }

    // BenchmarkFunction(TwoSpinCMatSpeedLevel0DefaultProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsDefault);
    // auto profResults = TwoSpinCMatSpeedLevel0DefaultProjects.GetProfile(); TwoSpinCMatSpeedLevel0DefaultProjects.ResetProfile();
    // prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    // std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 0, default projects) from baseline" << std::endl;
    // for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
    //   convertLayout(TwoSpinCMatSpeedLevel0DefaultProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    // }

    BenchmarkFunction(TwoSpinCMatSpeedLevel0FastProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsFast);
    profResults = TwoSpinCMatSpeedLevel0FastProjects.GetProfile(); TwoSpinCMatSpeedLevel0FastProjects.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 0, fast projects) from baseline" << std::endl;
    for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
      convertLayout(TwoSpinCMatSpeedLevel0FastProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    }

    // BenchmarkFunction(TwoSpinCMatSpeedLevel1DefaultProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsDefault);
    // profResults = TwoSpinCMatSpeedLevel1DefaultProjects.GetProfile(); TwoSpinCMatSpeedLevel1DefaultProjects.ResetProfile();
    // prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    // std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 1, default projects) from baseline" << std::endl;
    // for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
    //   convertLayout(TwoSpinCMatSpeedLevel1DefaultProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    // }

    BenchmarkFunction(TwoSpinCMatSpeedLevel1FastProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsFast);
    profResults = TwoSpinCMatSpeedLevel1FastProjects.GetProfile(); TwoSpinCMatSpeedLevel1FastProjects.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 1, fast projects) from baseline" << std::endl;
    for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
      convertLayout(TwoSpinCMatSpeedLevel1FastProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    }

    // BenchmarkFunction(TwoSpinCMatSpeedLevel2DefaultProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsDefault);
    // profResults = TwoSpinCMatSpeedLevel2DefaultProjects.GetProfile(); TwoSpinCMatSpeedLevel2DefaultProjects.ResetProfile();
    // prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    // std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 2, default projects) from baseline" << std::endl;
    // for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
    //   convertLayout(TwoSpinCMatSpeedLevel2DefaultProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    // }

    BenchmarkFunction(TwoSpinCMatSpeedLevel2FastProjects.CoarsenOperator, flop, byte, nIterOne, FGrid, LinOp, TwoSpinAggsFast);
    profResults = TwoSpinCMatSpeedLevel2FastProjects.GetProfile(); TwoSpinCMatSpeedLevel2FastProjects.ResetProfile();
    prettyPrintProfiling("", profResults, profResults["CoarsenOperator.Total"].t, false);
    std::cout << GridLogMessage << "Deviations of two-spin layout (speed level 2, fast projects) from baseline" << std::endl;
    for(int p = 0; p < BaselineCMat.geom.npoint; ++p) {
      convertLayout(TwoSpinCMatSpeedLevel2FastProjects.Y_[p], CoarseLFBaselineTmp); printDeviationFromReference(tol, BaselineCMat.A[p], CoarseLFBaselineTmp);
    }
  }
#endif

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

    double flop = CVolume * ((nStencil * (8 * CSiteElems * CSiteElems - 2 * CSiteElems) + nAccum * 2 * CSiteElems) + 8 * CSiteElems);
    double byte = CVolume * ((nStencil * (CSiteElems * CSiteElems + CSiteElems) + CSiteElems) + CSiteElems) * sizeof(Complex);

    BenchmarkFunction(UpstreamCMat.M,           flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMatSpeedLevel2.M, flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

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

    BenchmarkFunction(UpstreamCMat.Mdag,           flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMatSpeedLevel2.Mdag, flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

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

    BenchmarkFunction(UpstreamCMat.Mdir,           flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut, 2, 1);
    BenchmarkFunction(TwoSpinCMatSpeedLevel2.Mdir, flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut,  2, 1);

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

    BenchmarkFunction(UpstreamCMat.Mdiag,           flop, byte, nIter, CoarseVecUpstreamIn, CoarseVecUpstreamOut);
    BenchmarkFunction(TwoSpinCMatSpeedLevel2.Mdiag, flop, byte, nIter, CoarseVecTwospinIn,  CoarseVecTwospinOut);

    convertLayout(CoarseVecTwospinOut, CoarseVecUpstreamTmp);
    printDeviationFromReference(tol, CoarseVecUpstreamOut, CoarseVecUpstreamTmp);
  }

  Grid_finalize();
}
