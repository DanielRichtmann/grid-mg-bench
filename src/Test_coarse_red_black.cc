/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/Test_coarse_red_black.cc

    Copyright (C) 2015-2019

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
#include <Benchmark_helpers.h>

using namespace Grid;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::Rework;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_coarse_red_black
#ifndef NBASIS
#define NBASIS 40
#endif

// NOTE: The tests in this file are written in analogy to
// - tests/core/Test_wilson_even_odd.cc
// - tests/core/Test_wilson_clover.cc

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  const int        nBasis = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int        nB     = nBasis/2;
  Coordinate blockSize    = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({2, 2, 2, 2}));
  bool       doClover     = readFromCommandLineToggle(&argc, &argv, "--doclover");
  bool       fastProjects = readFromCommandLineToggle(&argc, &argv, "--fastprojects");
  int        speedLevel   = readFromCommandLineInt(&argc, &argv, "--speedlevel", 0);
  // clang-format on

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian*         FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian*         CGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);
  GridRedBlackCartesian* CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "CGrid:" << std::endl; CGrid->show_decomposition();
  std::cout << GridLogMessage << "FrbGrid:" << std::endl; FrbGrid->show_decomposition();
  std::cout << GridLogMessage << "CrbGrid:" << std::endl; CrbGrid->show_decomposition();

  GridParallelRNG FPRNG(FGrid);
  GridParallelRNG CPRNG(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  FPRNG.SeedFixedIntegers(seeds);
  CPRNG.SeedFixedIntegers(seeds);

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG, Umu);

  RealD checkTolerance = 1e-15;

  RealD                                               mass = -0.1;
  RealD                                               csw  = 1.0;
  WilsonFermionR                                      Dw(Umu, *FGrid, *FrbGrid, mass);
  WilsonCloverFermionR                                Dwc(Umu, *FGrid, *FrbGrid, mass, csw, csw);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDw(Dw);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDwc(Dwc);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion>* MdagMOp = nullptr;

  if(doClover) {
    MdagMOp = &MdagMOpDwc;
    std::cout << "Running tests for clover fermions" << std::endl;
  } else {
    MdagMOp = &MdagMOpDw;
    std::cout << "Running tests for wilson fermions" << std::endl;
  }

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

#if defined(USE_NEW_COARSENING)
  typedef CoarseningPolicy<LatticeFermion, nB, 1>                OneSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 2>                TwoSpinCoarseningPolicy;
  typedef CoarseningPolicy<LatticeFermion, nB, 4>                FourSpinCoarseningPolicy;
  typedef MGBasisVectors<LatticeFermion>                         BasisVectors;
  typedef Grid::Rework::Aggregation<TwoSpinCoarseningPolicy>     Aggregates;
  typedef Grid::Rework::CoarsenedMatrix<TwoSpinCoarseningPolicy> CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField               CoarseVector;
#else
  typedef MGBasisVectors<LatticeFermion>                                        BasisVectors;
  typedef Grid::Baseline::Aggregation<vSpinColourVector, vTComplex, nBasis>     Aggregates;
  typedef Grid::Baseline::CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> CoarseDiracMatrix;
  typedef CoarseDiracMatrix::CoarseVector                                       CoarseVector;
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

#if defined(USE_NEW_COARSENING)
  Aggregates Aggs(CGrid, FGrid, 0, fastProjects);
  for(int i = 0; i < Aggs.Subspace().size(); ++i) random(FPRNG, Aggs.Subspace()[i]);
  Aggs.Orthogonalise(1, 1); // check orthogonality, 1 pass of GS
#else
  Aggregates Aggs(CGrid, FGrid, 0);
  Aggs.CreateSubspaceRandom(FPRNG);
  performChiralDoubling(Aggs.subspace);
  Aggs.Orthogonalise(1, 1); // check orthogonality, 1 pass of GS
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                  Setup of CoarsenedMatrix and Operator                  //
  /////////////////////////////////////////////////////////////////////////////

  const int hermitian = 0;
#if defined(USE_NEW_COARSENING)
  CoarseDiracMatrix Dc(*CGrid, *CrbGrid, speedLevel, hermitian); // test 0, 1, 2 for first param
#else
  CoarseDiracMatrix Dc(*CGrid, *CrbGrid, hermitian);
#endif
  Dc.CoarsenOperator(FGrid, *MdagMOp, Aggs);

  MdagMLinearOperator<CoarseDiracMatrix, CoarseVector> MdagMOp_Dc(Dc);

  /////////////////////////////////////////////////////////////////////////////
  //                              Start of tests                             //
  /////////////////////////////////////////////////////////////////////////////

  CoarseVector src(CGrid); random(CPRNG, src);

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Testing that Dhop + Mdiag = Munprec" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector phi(CGrid);   phi = Zero();
    CoarseVector chi(CGrid);   chi = Zero();
    CoarseVector res(CGrid);   res = Zero();
    CoarseVector ref(CGrid);   ref = Zero();
    CoarseVector diff(CGrid); diff = Zero();

    Dc.Mdiag(src, phi);          std::cout << GridLogMessage << "Applied Mdiag" << std::endl;
    Dc.Dhop(src, chi, DaggerNo); std::cout << GridLogMessage << "Applied Dhop"  << std::endl;
    Dc.M(src, ref);              std::cout << GridLogMessage << "Applied M"     << std::endl;
    // clang-format on

    res = phi + chi;

    diff = ref - res;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(Munprec), norm2(Dhop + Mdiag), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(res) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Testing that Meo + Moe = Dhop" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector src_e(CrbGrid); src_e = Zero();
    CoarseVector src_o(CrbGrid); src_o = Zero();
    CoarseVector r_e(CrbGrid);     r_e = Zero();
    CoarseVector r_o(CrbGrid);     r_o = Zero();
    CoarseVector r_eo(CGrid);     r_eo = Zero();
    CoarseVector ref(CGrid);       ref = Zero();
    CoarseVector diff(CGrid);     diff = Zero();
    // clang-format on

    pickCheckerboard(Even, src_e, src);
    pickCheckerboard(Odd, src_o, src);

    // clang-format off
    Dc.Meooe(src_e, r_o);        std::cout << GridLogMessage << "Applied Meo"  << std::endl;
    Dc.Meooe(src_o, r_e);        std::cout << GridLogMessage << "Applied Moe"  << std::endl;
    Dc.Dhop(src, ref, DaggerNo); std::cout << GridLogMessage << "Applied Dhop" << std::endl;
    // clang-format on

    setCheckerboard(r_eo, r_o);
    setCheckerboard(r_eo, r_e);

    diff = ref - r_eo;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(Dhop), norm2(Meo + Moe), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(r_eo) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test |(Im(v^dag M^dag M v)| = 0" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector tmp(CGrid); tmp = Zero();
    CoarseVector phi(CGrid); phi = Zero();

    Dc.M(src, tmp);    std::cout << GridLogMessage << "Applied M"    << std::endl;
    Dc.Mdag(tmp, phi); std::cout << GridLogMessage << "Applied Mdag" << std::endl;
    // clang-format on

    std::cout << GridLogMessage << "src = " << norm2(src) << " tmp = " << norm2(tmp) << " phi = " << norm2(phi) << std::endl;

    ComplexD dot = innerProduct(src, phi);

    auto relDev = std::abs(imag(dot)) / std::abs(real(dot));
    std::cout << GridLogMessage << "Re(v^dag M^dag M v), Im(v^dag M^dag M v), rel.deviation: "
              << real(dot) << " " << imag(dot) << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test |(Im(v^dag Mooee^dag Mooee v)| = 0 (full grid)" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector tmp(CGrid); tmp = Zero();
    CoarseVector phi(CGrid); phi = Zero();

    Dc.Mooee(src, tmp);    std::cout << GridLogMessage << "Applied Mooee"    << std::endl;
    Dc.MooeeDag(tmp, phi); std::cout << GridLogMessage << "Applied MooeeDag" << std::endl;
    // clang-format on

    ComplexD dot = innerProduct(src, phi);

    auto relDev = std::abs(imag(dot)) / std::abs(real(dot));
    std::cout << GridLogMessage << "Re(v^dag Mooee^dag Mooee v), Im(v^dag Mooee^dag Mooee v), rel.deviation: "
              << real(dot) << " " << imag(dot) << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test MooeeInv Mooee = 1 (full grid)" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector tmp(CGrid);   tmp = Zero();
    CoarseVector phi(CGrid);   phi = Zero();
    CoarseVector diff(CGrid); diff = Zero();

    Dc.Mooee(src, tmp);    std::cout << GridLogMessage << "Applied Mooee"    << std::endl;
    Dc.MooeeInv(tmp, phi); std::cout << GridLogMessage << "Applied MooeeInv" << std::endl;
    // clang-format on

    diff        = src - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(src);
    std::cout << GridLogMessage << "norm2(src), norm2(MooeeInv Mooee src), abs. deviation, rel. deviation: "
              << norm2(src) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test Ddagger is the dagger of D by requiring" << std::endl;
    std::cout << GridLogMessage << " < phi | Meo | chi > * = < chi | Meo^dag| phi>" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector phi(CGrid); random(CPRNG, phi);
    CoarseVector chi(CGrid); random(CPRNG, chi);
    CoarseVector chi_e(CrbGrid);   chi_e = Zero();
    CoarseVector chi_o(CrbGrid);   chi_o = Zero();
    CoarseVector dchi_e(CrbGrid); dchi_e = Zero();
    CoarseVector dchi_o(CrbGrid); dchi_o = Zero();
    CoarseVector phi_e(CrbGrid);   phi_e = Zero();
    CoarseVector phi_o(CrbGrid);   phi_o = Zero();
    CoarseVector dphi_e(CrbGrid); dphi_e = Zero();
    CoarseVector dphi_o(CrbGrid); dphi_o = Zero();
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, phi_e, phi);
    pickCheckerboard(Odd, phi_o, phi);

    // clang-format off
    Dc.Meooe(chi_e, dchi_o);    std::cout << GridLogMessage << "Applied Meo"    << std::endl;
    Dc.Meooe(chi_o, dchi_e);    std::cout << GridLogMessage << "Applied Moe"    << std::endl;
    Dc.MeooeDag(phi_e, dphi_o); std::cout << GridLogMessage << "Applied MeoDag" << std::endl;
    Dc.MeooeDag(phi_o, dphi_e); std::cout << GridLogMessage << "Applied MoeDag" << std::endl;
    // clang-format on

    ComplexD phiDchi_e = innerProduct(phi_e, dchi_e);
    ComplexD phiDchi_o = innerProduct(phi_o, dchi_o);
    ComplexD chiDphi_e = innerProduct(chi_e, dphi_e);
    ComplexD chiDphi_o = innerProduct(chi_o, dphi_o);

    std::cout << GridLogDebug << "norm dchi_e = " << norm2(dchi_e) << " norm dchi_o = " << norm2(dchi_o) << " norm dphi_e = " << norm2(dphi_e)
              << " norm dphi_o = " << norm2(dphi_e) << std::endl;

    std::cout << GridLogMessage << "e " << phiDchi_e << " " << chiDphi_e << std::endl;
    std::cout << GridLogMessage << "o " << phiDchi_o << " " << chiDphi_o << std::endl;

    std::cout << GridLogMessage << "phiDchi_e - conj(chiDphi_o) " << phiDchi_e - conj(chiDphi_o) << std::endl;
    std::cout << GridLogMessage << "phiDchi_o - conj(chiDphi_e) " << phiDchi_o - conj(chiDphi_e) << std::endl;
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test MooeeInv Mooee = 1 (checkerboards separately)" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector chi(CGrid);   random(CPRNG, chi);
    CoarseVector tmp(CGrid);   tmp = Zero();
    CoarseVector phi(CGrid);   phi = Zero();
    CoarseVector diff(CGrid); diff = Zero();
    CoarseVector chi_e(CrbGrid); chi_e = Zero();
    CoarseVector chi_o(CrbGrid); chi_o = Zero();
    CoarseVector phi_e(CrbGrid); phi_e = Zero();
    CoarseVector phi_o(CrbGrid); phi_o = Zero();
    CoarseVector tmp_e(CrbGrid); tmp_e = Zero();
    CoarseVector tmp_o(CrbGrid); tmp_o = Zero();
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, tmp_e, tmp);
    pickCheckerboard(Odd, tmp_o, tmp);

    // clang-format off
    Dc.Mooee(chi_e, tmp_e);    std::cout << GridLogMessage << "Applied Mee"    << std::endl;
    Dc.MooeeInv(tmp_e, phi_e); std::cout << GridLogMessage << "Applied MeeInv" << std::endl;
    Dc.Mooee(chi_o, tmp_o);    std::cout << GridLogMessage << "Applied Moo"    << std::endl;
    Dc.MooeeInv(tmp_o, phi_o); std::cout << GridLogMessage << "Applied MooInv" << std::endl;
    // clang-format on

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    diff = chi - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(chi);
    std::cout << GridLogMessage << "norm2(chi), norm2(MeeInv Mee chi), abs. deviation, rel. deviation: "
              << norm2(chi) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Test MooeeDag MooeeInvDag = 1 (checkerboards separately)" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector chi(CGrid);   random(CPRNG, chi);
    CoarseVector tmp(CGrid);   tmp = Zero();
    CoarseVector phi(CGrid);   phi = Zero();
    CoarseVector diff(CGrid); diff = Zero();
    CoarseVector chi_e(CrbGrid); chi_e = Zero();
    CoarseVector chi_o(CrbGrid); chi_o = Zero();
    CoarseVector phi_e(CrbGrid); phi_e = Zero();
    CoarseVector phi_o(CrbGrid); phi_o = Zero();
    CoarseVector tmp_e(CrbGrid); tmp_e = Zero();
    CoarseVector tmp_o(CrbGrid); tmp_o = Zero();
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, tmp_e, tmp);
    pickCheckerboard(Odd, tmp_o, tmp);

    // clang-format off
    Dc.MooeeDag(chi_e, tmp_e);    std::cout << GridLogMessage << "Applied MeeDag"    << std::endl;
    Dc.MooeeInvDag(tmp_e, phi_e); std::cout << GridLogMessage << "Applied MeeInvDag" << std::endl;
    Dc.MooeeDag(chi_o, tmp_o);    std::cout << GridLogMessage << "Applied MooDag"    << std::endl;
    Dc.MooeeInvDag(tmp_o, phi_o); std::cout << GridLogMessage << "Applied MooInvDag" << std::endl;
    // clang-format on

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    diff = chi - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(chi);
    std::cout << GridLogMessage << "norm2(chi), norm2(MeeDag MeeInvDag chi), abs. deviation, rel. deviation: "
              << norm2(chi) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Testing that Meo + Moe + Moo + Mee = Munprec" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector chi(CGrid);   chi = Zero();
    CoarseVector phi(CGrid);   phi = Zero();
    CoarseVector ref(CGrid);   ref = Zero();
    CoarseVector diff(CGrid); diff = Zero();
    CoarseVector src_e(CrbGrid); src_e = Zero();
    CoarseVector src_o(CrbGrid); src_o = Zero();
    CoarseVector phi_e(CrbGrid); phi_e = Zero();
    CoarseVector phi_o(CrbGrid); phi_o = Zero();
    CoarseVector chi_e(CrbGrid); chi_e = Zero();
    CoarseVector chi_o(CrbGrid); chi_o = Zero();
    // clang-format on

    pickCheckerboard(Even, src_e, src);
    pickCheckerboard(Odd, src_o, src);
    pickCheckerboard(Even, phi_e, phi);
    pickCheckerboard(Odd, phi_o, phi);
    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);

    // M phi = (Mooee src_e + Meooe src_o , Mooee src_o + Meooe src_e)

    Dc.M(src, ref); // Reference result from the unpreconditioned operator

    // EO matrix
    // clang-format off
    Dc.Mooee(src_e, chi_e); std::cout << GridLogMessage << "Applied Mee" << std::endl;
    Dc.Mooee(src_o, chi_o); std::cout << GridLogMessage << "Applied Moo" << std::endl;
    Dc.Meooe(src_o, phi_e); std::cout << GridLogMessage << "Applied Moe" << std::endl;
    Dc.Meooe(src_e, phi_o); std::cout << GridLogMessage << "Applied Meo" << std::endl;
    // clang-format on

    phi_o += chi_o;
    phi_e += chi_e;

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    std::cout << GridLogDebug << "norm phi_e = " << norm2(phi_e) << " norm phi_o = " << norm2(phi_o) << " norm phi = " << norm2(phi) << std::endl;

    diff = ref - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(Dunprec), norm2(Deoprec), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
    if(relDev >= checkTolerance) abort();
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Testing that MpcDagMpc is hermitian" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    // clang-format off
    CoarseVector phi(CGrid); random(CPRNG, phi);
    CoarseVector chi(CGrid); random(CPRNG, chi);
    CoarseVector chi_e(CrbGrid);   chi_e = Zero();
    CoarseVector chi_o(CrbGrid);   chi_o = Zero();
    CoarseVector dchi_e(CrbGrid); dchi_e = Zero();
    CoarseVector dchi_o(CrbGrid); dchi_o = Zero();
    CoarseVector phi_e(CrbGrid);   phi_e = Zero();
    CoarseVector phi_o(CrbGrid);   phi_o = Zero();
    CoarseVector dphi_e(CrbGrid); dphi_e = Zero();
    CoarseVector dphi_o(CrbGrid); dphi_o = Zero();
    RealD t1, t2;
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, phi_e, phi);
    pickCheckerboard(Odd, phi_o, phi);

    NonHermitianSchurDiagMooeeOperator<CoarseDiracMatrix,CoarseVector> NonHermOpEO(Dc);

    // clang-format off
    NonHermOpEO.MpcDagMpc(chi_e, dchi_e, t1, t2); std::cout << GridLogMessage << "Applied MpcDagMpc to chi_e" << std::endl;
    NonHermOpEO.MpcDagMpc(chi_o, dchi_o, t1, t2); std::cout << GridLogMessage << "Applied MpcDagMpc to chi_o" << std::endl;
    NonHermOpEO.MpcDagMpc(phi_e, dphi_e, t1, t2); std::cout << GridLogMessage << "Applied MpcDagMpc to phi_e" << std::endl;
    NonHermOpEO.MpcDagMpc(phi_o, dphi_o, t1, t2); std::cout << GridLogMessage << "Applied MpcDagMpc to phi_o" << std::endl;
    // clang-format on

    ComplexD phiDchi_e = innerProduct(phi_e, dchi_e);
    ComplexD phiDchi_o = innerProduct(phi_o, dchi_o);
    ComplexD chiDphi_e = innerProduct(chi_e, dphi_e);
    ComplexD chiDphi_o = innerProduct(chi_o, dphi_o);

    std::cout << GridLogDebug << "norm dchi_e = " << norm2(dchi_e) << " norm dchi_o = " << norm2(dchi_o) << " norm dphi_e = " << norm2(dphi_e)
              << " norm dphi_o = " << norm2(dphi_e) << std::endl;

    std::cout << GridLogMessage << "e " << phiDchi_e << " " << chiDphi_e << std::endl;
    std::cout << GridLogMessage << "o " << phiDchi_o << " " << chiDphi_o << std::endl;

    std::cout << GridLogMessage << "phiDchi_e - conj(chiDphi_e) " << phiDchi_e - conj(chiDphi_e) << std::endl;
    std::cout << GridLogMessage << "phiDchi_o - conj(chiDphi_o) " << phiDchi_o - conj(chiDphi_o) << std::endl;
  }

  {
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;
    std::cout << GridLogMessage << "Comparing EO solve is with unprec one" << std::endl;
    std::cout << GridLogMessage << "***************************************************************************" << std::endl;

    GridStopWatch Timer;

    RealD   solverTolerance = 1e-12;
    Integer maxIter         = 10000;
    Integer restartLength   = 25;

    // clang-format off
    CoarseVector resultCG(CGrid);           resultCG = Zero();
    CoarseVector resultRBCG(CGrid);       resultRBCG = Zero();
    CoarseVector resultGMRES(CGrid);     resultGMRES = Zero();
    CoarseVector resultRBGMRES(CGrid); resultRBGMRES = Zero();
    // clang-format on

    ConjugateGradient<CoarseVector> CG(solverTolerance, maxIter);
    SchurRedBlackDiagMooeeSolve<CoarseVector> RBCG(CG);
    GeneralisedMinimalResidual<CoarseVector> GMRES(solverTolerance, maxIter, restartLength);
    NonHermitianSchurRedBlackDiagMooeeSolve<CoarseVector> RBGMRES(GMRES);

    // clang-format off
    Timer.Reset(); Timer.Start();
    CG(MdagMOp_Dc, src, resultCG);
    Timer.Stop(); std::cout << "CG took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    RBCG(Dc, src, resultRBCG);
    Timer.Stop(); std::cout << "RBCG took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    GMRES(MdagMOp_Dc, src, resultGMRES);
    Timer.Stop(); std::cout << "GMRES took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    RBGMRES(Dc, src, resultRBGMRES);
    Timer.Stop(); std::cout << "RBGMRES took " << Timer.Elapsed() << std::endl;
    // clang-format on
  }

  Grid_finalize();
}
