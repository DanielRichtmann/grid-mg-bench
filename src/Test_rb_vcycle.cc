/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./src/Test_rb_vcycle

    Copyright (C) 2015 - Current Year

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>

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
#include <Benchmark_helpers.h>
#include <Helpers.h>
#include <VCycleRework.h>


#define TWO_LEVELS
#define THREE_LEVELS
#define SINGLE_APPLY
#define STANDALONE
#define PRECONDITIONER
#define NBASIS 12


using namespace Grid;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::Rework;


template<class FineObject, class CComplex, int nbasis>
class NonHermitianNullVectorSetup {
public: // type definitions
  typedef Aggregation<FineObject, CComplex, nbasis> Aggregates;
  typedef typename Aggregates::FineField            FineField;
  typedef LinearOperatorBase<FineField>             FineOperator;
  typedef LinearFunction<FineField>                 FineSolver;

public: // sanity checks
  static_assert(nbasis % 2 == 0, "Must be even");

public: // data members
  FineOperator&    FineOperator_;
  GridParallelRNG& RNG_;
  static const int nb = nbasis / 2;

public: // constructors
  NonHermitianNullVectorSetup(FineOperator& FineOp, GridParallelRNG& RNG)
    : FineOperator_(FineOp), RNG_(RNG) {}

  void operator()(Aggregates& Aggs, FineSolver& FineSolver, bool fromRandom=true) {
    // fromRandom = true/false -> inital setup/setup update
    assert(Aggs.subspace.size() == nbasis);
    FineField null(Aggs.FineGrid); null.Checkerboard() = Aggs.subspace[0].Checkerboard(); null = Zero();
    const int nb = nbasis/2;

    if(fromRandom) for(int n=0; n<nb; n++) gaussian(RNG_, Aggs.subspace[n]);
    else undoChiralDoublingG5C(Aggs.subspace);

    for(int n=0; n<nb; n++) FineSolver(null, Aggs.subspace[n]);

    basisOrthonormalize(Aggs.subspace, true); // gobal orthonormalization

    performChiralDoublingG5C(Aggs.subspace);

    // block-wise orthonormalization (2 passes)
    typename Aggregates::CoarseScalar ip(Aggs.CoarseGrid);
    blockOrthogonalise(ip, Aggs.subspace);
    blockOrthogonalise(ip, Aggs.subspace);
  }
};


void printHeader(std::string const& message) {
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "****************************************************************" << std::endl;
  std::cout << GridLogMessage << "******* Starting test \"" << message << "\"" << std::endl;
  std::cout << GridLogMessage << "****************************************************************" << std::endl;
}
void printFooter(std::string const& message) {
  std::cout << GridLogMessage << "****************************************************************" << std::endl;
  std::cout << GridLogMessage << "******* Test \"" << message << "\" passed" << std::endl;
  std::cout << GridLogMessage << "****************************************************************" << std::endl;
}


template<int nbasis, int nbasisc>
void runTest(int* argc, char*** argv) {
  Grid_init(argc, argv);
  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  const int nb      = nbasis / 2;
  const int nbc     = nb / nbasisc;

  Coordinate blockSize = readFromCommandLineCoordinate(argc, argv, "--blocksize", Coordinate({2, 2, 2, 2}));
  Coordinate cblockSize = readFromCommandLineCoordinate(argc, argv, "--cblocksize", Coordinate({2, 2, 2, 2}));

  std::cout << GridLogMessage << "Run with nbasis = " << nbasis << " -> nb = " << nb << std::endl;
  std::cout << GridLogMessage << "Run with nbasisc = " << nbasisc << " -> nbc = " << nbc << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt  = calcCoarseLattSize(GridDefaultLatt(), blockSize);
  Coordinate cclatt = calcCoarseLattSize(clatt, cblockSize);

  GridCartesian*         FGrid    = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* FrbGrid  = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);
  GridCartesian*         CGrid    = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* CrbGrid  = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);
#if defined(THREE_LEVELS)
  GridCartesian*         CCGrid   = SpaceTimeGrid::makeFourDimGrid(cclatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* CCrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CCGrid);
#endif

  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "FrbGrid:" << std::endl; FrbGrid->show_decomposition();
  std::cout << GridLogMessage << "CGrid:" << std::endl; CGrid->show_decomposition();
  std::cout << GridLogMessage << "CrbGrid:" << std::endl; CrbGrid->show_decomposition();
#if defined(THREE_LEVELS)
  std::cout << GridLogMessage << "CCGrid:" << std::endl; CCGrid->show_decomposition();
  std::cout << GridLogMessage << "CCrbGrid:" << std::endl; CCrbGrid->show_decomposition();
#endif

  GridParallelRNG FPRNG(FGrid);
  GridParallelRNG FRBPRNG(FrbGrid);
  GridParallelRNG CPRNG(CGrid);
  GridParallelRNG CRBPRNG(CrbGrid);
#if defined(THREE_LEVELS)
  GridParallelRNG CCPRNG(CCGrid);
#endif

  std::vector<int> seeds({1, 2, 3, 4});

  FPRNG.SeedFixedIntegers(seeds);
  FRBPRNG.SeedFixedIntegers(seeds);
  CPRNG.SeedFixedIntegers(seeds);
  CRBPRNG.SeedFixedIntegers(seeds);
#if defined(THREE_LEVELS)
  CPRNG.SeedFixedIntegers(seeds);
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  // fine matrix
  typedef WilsonCloverFermionR               WilsonCloverOperator;
  typedef WilsonCloverFermionR::FermionField FermionField;

  // first mg level
  typedef Aggregation<vSpinColourVector, vTComplex, nbasis>                 Aggregates;
  typedef CoarsenedMatrix<vSpinColourVector, vTComplex, nbasis>             CoarseOperator;
  typedef typename CoarseOperator::CoarseVector                             CoarseVector;
  typedef typename CoarseOperator::siteVector                               CoarseSiteVector;
  typedef NonHermitianNullVectorSetup<vSpinColourVector, vTComplex, nbasis> NullVectorSetup;
  typedef NonHermitianVCyclePreconditioner<vSpinColourVector, vTComplex, nbasis, LinearFunction<CoarseVector>>
    VCyclePreconditioner;
  typedef NonHermitianVCyclePreconditionerTest<vSpinColourVector,
                                               vTComplex,
                                               nbasis,
                                               LinearFunction<CoarseVector>>
    VCyclePreconditionerTest;

  // second mg level
  typedef Aggregation<CoarseSiteVector, iScalar<vTComplex>, nbasisc>                 CoarseAggregates;
  typedef CoarsenedMatrix<CoarseSiteVector, iScalar<vTComplex>, nbasisc>             CoarseCoarseOperator;
  typedef typename CoarseCoarseOperator::CoarseVector                                CoarseCoarseVector;
  typedef NonHermitianNullVectorSetup<CoarseSiteVector, iScalar<vTComplex>, nbasisc> CoarseNullVectorSetup;
  typedef NonHermitianVCyclePreconditioner<CoarseSiteVector,
                                           iScalar<vTComplex>,
                                           nbasisc,
                                           LinearFunction<CoarseCoarseVector>>
    CoarseVCyclePreconditioner;
  typedef NonHermitianVCyclePreconditionerTest<CoarseSiteVector,
                                               iScalar<vTComplex>,
                                               nbasisc,
                                               LinearFunction<CoarseCoarseVector>>
    CoarseVCyclePreconditionerTest;

  /////////////////////////////////////////////////////////////////////////////
  //                   Setup of Dirac Matrix and Operators                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG, Umu);

  RealD mass = -0.10; RealD csw = 1.25;

  WilsonCloverOperator                                                 Dwc(Umu, *FGrid, *FrbGrid, mass, csw, csw);
  MdagMLinearOperator<WilsonCloverOperator, FermionField>              MdagMOp_Dwc(Dwc);
  NonHermitianSchurDiagTwoOperator<WilsonCloverOperator, FermionField> DiagTwoOp_Dwc(Dwc);

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  Aggregates                                          Aggs(CGrid, FGrid, 0);
  BiCGSTAB<FermionField>                              SetupSlv(1e-5, 500, false);
  SolverWrapper<FermionField>                         SetupSolver(MdagMOp_Dwc, SetupSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> SetupSchurRBSlv(SetupSlv, false, true); // forward initial guess
  SchurSolverWrapper<FermionField>                    SetupSchurRBSolver(Dwc, SetupSchurRBSlv);
  NullVectorSetup                                     Setup(MdagMOp_Dwc, FPRNG);
  // Setup(Aggs, SetupSolver); // use full operator to get null vecs of full operator
  Setup(Aggs, SetupSchurRBSolver); // use schur rb operator to get null vecs of full operator

  /////////////////////////////////////////////////////////////////////////////
  //                     Setup of Aggregation in eo space                    //
  /////////////////////////////////////////////////////////////////////////////

  Aggregates                  RBAggs(CrbGrid, FrbGrid, Odd);
  BiCGSTAB<FermionField>      SetupDirectRBSlv(1e-5, 500, false);
  SolverWrapper<FermionField> SetupDirectRBSolver(DiagTwoOp_Dwc, SetupDirectRBSlv);
  NullVectorSetup             RBSetup(DiagTwoOp_Dwc, FRBPRNG);

  // dedicated setup for the rb prec operator <- is this correct?
  // RBSetup(RBAggs, SetupDirectRBSolver);

  // reuse the setup from the unprec operator
  for(int n=0; n<Aggs.subspace.size(); n++) {
    pickCheckerboard(Odd, RBAggs.subspace[n], Aggs.subspace[n]);
  }

  /////////////////////////////////////////////////////////////////////////////
  //                  Setup of CoarsenedMatrix and Operators                 //
  /////////////////////////////////////////////////////////////////////////////

  const int hermitian = 0;
  CoarseOperator Dc(*CGrid, *CrbGrid, hermitian); Dc.CoarsenOperator(FGrid, MdagMOp_Dwc, Aggs);
  MdagMLinearOperator<CoarseOperator, CoarseVector>              MdagMOp_Dc(Dc);
  NonHermitianSchurDiagTwoOperator<CoarseOperator, CoarseVector> DiagTwoOp_Dc(Dc);

  /////////////////////////////////////////////////////////////////////////////
  //              Setup of constituents of VCycle preconditioner             //
  /////////////////////////////////////////////////////////////////////////////

  // unpreconditioned solvers + their wrappers
  MinimalResidual<FermionField>                       PreSmoothSlv(0.1, 4, 1.1, false);
  MinimalResidual<FermionField>                       PostSmoothSlv(0.1, 4, 1.1, false);
  GeneralisedMinimalResidual<CoarseVector>            CoarseSlv(0.1, 200, 10, false);
  SolverWrapper<FermionField>                         PreSmoothSolver(MdagMOp_Dwc, PreSmoothSlv);
  SolverWrapper<FermionField>                         PostSmoothSolver(MdagMOp_Dwc, PostSmoothSlv);
  SolverWrapper<CoarseVector>                         CoarseSolver(MdagMOp_Dc, CoarseSlv);

  // eo preconditioned schur solvers + their wrappers
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> PreSmoothSchurRBSlv(PreSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> PostSmoothSchurRBSlv(PostSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector> CoarseSchurRBSlv(CoarseSlv);
  SchurSolverWrapper<FermionField>                    PreSmoothSchurRBSolver(Dwc, PreSmoothSchurRBSlv);
  SchurSolverWrapper<FermionField>                    PostSmoothSchurRBSolver(Dwc, PostSmoothSchurRBSlv);
  SchurSolverWrapper<CoarseVector>                    CoarseSchurRBSolver(Dc, CoarseSchurRBSlv);

  // eo preconditioned direct solvers + their wrappers
  SolverWrapper<FermionField> PreSmoothDirectRBSolver(DiagTwoOp_Dwc, PreSmoothSlv);
  SolverWrapper<FermionField> PostSmoothDirectRBSolver(DiagTwoOp_Dwc, PostSmoothSlv);
  SolverWrapper<CoarseVector> CoarseDirectRBSolver(DiagTwoOp_Dc, CoarseSlv);

#if defined(THREE_LEVELS)
  /////////////////////////////////////////////////////////////////////////////
  //                       Setup of coarse Aggregation                       //
  /////////////////////////////////////////////////////////////////////////////

  CoarseAggregates                                    CoarseAggs(CCGrid, CGrid, 0);
  BiCGSTAB<CoarseVector>                              CoarseSetupSlv(1e-5, 500, false);
  SolverWrapper<CoarseVector>                         CoarseSetupSolver(MdagMOp_Dc, CoarseSetupSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector> CoarseSetupSchurRBSlv(
    CoarseSetupSlv, false, true); // forward initial guess
  SchurSolverWrapper<CoarseVector> CoarseSetupSchurRBSolver(Dc, CoarseSetupSchurRBSlv);
  CoarseNullVectorSetup            CoarseSetup(MdagMOp_Dc, CPRNG);
  // CoarseSetup(CoarseAggs, CoarseSetupSolver); // use full operator to get null vecs of full operator
  CoarseSetup(CoarseAggs,
              CoarseSetupSchurRBSolver); // use schur rb operator to get null vecs of full operator

  /////////////////////////////////////////////////////////////////////////////
  //                 Setup of coarse Aggregation in eo space                 //
  /////////////////////////////////////////////////////////////////////////////

  CoarseAggregates            CoarseRBAggs(CCrbGrid, CrbGrid, Odd);
  BiCGSTAB<CoarseVector>      CoarseSetupDirectRBSlv(1e-5, 500, false);
  SolverWrapper<CoarseVector> CoarseSetupDirectRBSolver(DiagTwoOp_Dc, CoarseSetupDirectRBSlv);
  CoarseNullVectorSetup       CoarseRBSetup(DiagTwoOp_Dc, CRBPRNG);

  // dedicated setup for the rb prec operator <- is this correct?
  // CoarseRBSetup(CoarseRBAggs, CoarseSetupDirectRBSolver);

  // reuse the setup from the unprec operator
  for(int n=0; n<CoarseAggs.subspace.size(); n++) {
    pickCheckerboard(Odd, CoarseRBAggs.subspace[n], CoarseAggs.subspace[n]);
  }

  /////////////////////////////////////////////////////////////////////////////
  //              Setup of coarse CoarsenedMatrix and Operators              //
  /////////////////////////////////////////////////////////////////////////////

  CoarseCoarseOperator Dcc(*CCGrid, *CCrbGrid, hermitian); Dcc.CoarsenOperator(CGrid, MdagMOp_Dc, CoarseAggs);
  MdagMLinearOperator<CoarseCoarseOperator, CoarseCoarseVector>              MdagMOp_Dcc(Dcc);
  NonHermitianSchurDiagTwoOperator<CoarseCoarseOperator, CoarseCoarseVector> DiagTwoOp_Dcc(Dcc);

  /////////////////////////////////////////////////////////////////////////////
  //          Setup of constituents of coarse VCycle preconditioner          //
  /////////////////////////////////////////////////////////////////////////////

  // coarse unpreconditioned solvers + their wrappers
  MinimalResidual<CoarseVector>                  CoarsePreSmoothSlv(0.1, 4, 1.1, false);
  MinimalResidual<CoarseVector>                  CoarsePostSmoothSlv(0.1, 4, 1.1, false);
  GeneralisedMinimalResidual<CoarseCoarseVector> CoarseCoarseSlv(0.1, 200, 10, false);
  SolverWrapper<CoarseVector>                    CoarsePreSmoothSolver(MdagMOp_Dc, CoarsePreSmoothSlv);
  SolverWrapper<CoarseVector>                    CoarsePostSmoothSolver(MdagMOp_Dc, CoarsePostSmoothSlv);
  SolverWrapper<CoarseCoarseVector>              CoarseCoarseSolver(MdagMOp_Dcc, CoarseCoarseSlv);

  // coarse eo preconditioned schur solvers + their wrappers
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector>       CoarsePreSmoothSchurRBSlv(CoarsePreSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector>       CoarsePostSmoothSchurRBSlv(CoarsePostSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseCoarseVector> CoarseCoarseSchurRBSlv(CoarseCoarseSlv);
  SchurSolverWrapper<CoarseVector>       CoarsePreSmoothSchurRBSolver(Dc, CoarsePreSmoothSchurRBSlv);
  SchurSolverWrapper<CoarseVector>       CoarsePostSmoothSchurRBSolver(Dc, CoarsePostSmoothSchurRBSlv);
  SchurSolverWrapper<CoarseCoarseVector> CoarseCoarseSchurRBSolver(Dcc, CoarseCoarseSchurRBSlv);

  // coarse eo preconditioned direct solvers + their wrappers
  SolverWrapper<CoarseVector> CoarsePreSmoothDirectRBSolver(DiagTwoOp_Dc, CoarsePreSmoothSlv);
  SolverWrapper<CoarseVector> CoarsePostSmoothDirectRBSolver(DiagTwoOp_Dc, CoarsePostSmoothSlv);
  SolverWrapper<CoarseCoarseVector> CoarseCoarseDirectRBSolver(DiagTwoOp_Dcc, CoarseCoarseSlv);
#endif

  /////////////////////////////////////////////////////////////////////////////
  //                        Setup fields for the tests                       //
  /////////////////////////////////////////////////////////////////////////////

  LatticeFermion src(FGrid);     gaussian(FPRNG, src);
  LatticeFermion psi(FGrid);     psi = Zero();
  LatticeFermion r(FGrid);       r = Zero();
  LatticeFermion src_o(FrbGrid); pickCheckerboard(Odd, src_o, src);
  LatticeFermion psi_o(FrbGrid); psi_o.Checkerboard() = Odd; psi_o = Zero();
  LatticeFermion r_o(FrbGrid);   r_o.Checkerboard()   = Odd; r_o   = Zero();

#define initializeFields(PSI, R) \
  { \
    (PSI) = Zero(); \
    (R)   = Zero(); \
  }

#define assertCorrect(OP, PSI, R, SRC, TOL) \
  { \
    (OP).Op((PSI), (R)); \
    sub((R), (R), (SRC)); \
    std::cout << GridLogMessage \
              << "L2-norms of fields:" \
              << " |psi| = "     << sqrt(norm2((PSI))) \
              << " |src| = "     << sqrt(norm2((SRC))) \
              << " |r| = "       << sqrt(norm2((R))) \
              << " |r|/|src| = " << sqrt(norm2((R)) / norm2((SRC))) \
              << " tol = "       << (TOL) \
              << std::endl; \
    assert(sqrt(norm2((R)) / norm2((SRC))) < (TOL)); \
  }

  /////////////////////////////////////////////////////////////////////////////
  //                                  Tests                                  //
  /////////////////////////////////////////////////////////////////////////////

#if defined(TWO_LEVELS) && defined(SINGLE_APPLY)
  {
    printHeader("apply a single 2lvl vcycle -- everything on full grid");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, MdagMOp_Dc, CoarseSolver, 0, 1);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 2lvl vcycle -- everything on full grid");
  }

  {
    printHeader("apply a single 2lvl vcycle -- schur smoother and coarse solver");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(Aggs,
                                MdagMOp_Dwc,
                                PreSmoothSchurRBSolver,
                                PostSmoothSchurRBSolver,
                                MdagMOp_Dc,
                                CoarseSchurRBSolver,
                                0,
                                1);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 2lvl vcycle -- schur smoother and coarse solver");
  }

  {
    printHeader("apply a single 2lvl vcycle -- schur around vcycle");
    initializeFields(psi, r);

    VCyclePreconditioner                                VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseDirectRBSolver,
                                0,
                                1);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField> WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    WrappedVCycleSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 2lvl vcycle -- schur around vcycle");
  }

  {
    printHeader("apply a single 2lvl vcycle -- everything in eo space");
    initializeFields(psi_o, r_o);

    VCyclePreconditioner VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseDirectRBSolver,
                                0,
                                1);
    VCycle(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 3.8e-1);
    printFooter("apply a single 2lvl vcycle -- everything in eo space");
  }
#endif

#if defined(TWO_LEVELS) && defined(STANDALONE)
  {
    printHeader("use the 2lvl vcycle as a standalone solver -- everything on full grid");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, MdagMOp_Dc, CoarseSolver, 0, 500, 5e-5);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 2lvl vcycle as a standalone solver -- everything on full grid");
  }

  {
    printHeader("use the 2lvl vcycle as a standalone solver -- schur smoother and coarse solver");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(Aggs,
                                MdagMOp_Dwc,
                                PreSmoothSchurRBSolver,
                                PostSmoothSchurRBSolver,
                                MdagMOp_Dc,
                                CoarseSchurRBSolver,
                                0,
                                500,
                                5e-5);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 2lvl vcycle as a standalone solver -- schur smoother and coarse solver");
  }

  {
    printHeader("use the 2lvl vcycle as a standalone solver -- schur around vcycle");
    initializeFields(psi, r);

    VCyclePreconditioner                                VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseDirectRBSolver,
                                0,
                                500,
                                5e-5);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField> WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    WrappedVCycleSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 2lvl vcycle as a standalone solver -- schur around vcycle");
  }

  {
    printHeader("use the 2lvl vcycle as a standalone solver -- everything in eo space");
    initializeFields(psi_o, r_o);

    VCyclePreconditioner VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseDirectRBSolver,
                                0,
                                500,
                                5e-5);
    VCycle(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 5.0e-5);
    printFooter("use the 2lvl vcycle as a standalone solver -- everything in eo space");
  }
#endif

#if defined(TWO_LEVELS) && defined(PRECONDITIONER)
  {
    printHeader("use the 2lvl vcycle as a preconditioner -- everything on full grid");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, MdagMOp_Dc, CoarseSolver, 0, 1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 2lvl vcycle as a preconditioner -- everything on full grid");
  }

  {
    printHeader("use the 2lvl vcycle as a preconditioner -- schur smoother and coarse solver");
    initializeFields(psi, r);

    VCyclePreconditioner                             VCycle(Aggs,
                                MdagMOp_Dwc,
                                PreSmoothSchurRBSolver,
                                PostSmoothSchurRBSolver,
                                MdagMOp_Dc,
                                CoarseSchurRBSolver,
                                0,
                                1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 2lvl vcycle as a preconditioner -- schur smoother and coarse solver");
  }

  {
    printHeader("use the 2lvl vcycle as a preconditioner -- schur around vcycle");
    initializeFields(psi, r);

    VCyclePreconditioner                                VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseDirectRBSolver,
                                0,
                                1);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField>                 WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, WrappedVCycleSchurRBSolver, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 2lvl vcycle as a preconditioner -- schur around vcycle");
  }

  {
    printHeader("use the 2lvl vcycle as a preconditioner -- everything in eo space");
    initializeFields(psi_o, r_o);

    VCyclePreconditioner VCycle(
      RBAggs, DiagTwoOp_Dwc, PreSmoothDirectRBSolver, PostSmoothDirectRBSolver, DiagTwoOp_Dc, CoarseDirectRBSolver, 0, 1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(DiagTwoOp_Dwc, OuterSlv);
    OuterSolver(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 1.0e-5);
    printFooter("use the 2lvl vcycle as a preconditioner -- everything in eo space");
  }

  {
    printHeader("use the 2lvl vcycle as a preconditioner -- schur around outer solver");
    initializeFields(psi, r);

    VCyclePreconditioner VCycle(RBAggs, DiagTwoOp_Dwc, PreSmoothDirectRBSolver, PostSmoothDirectRBSolver, DiagTwoOp_Dc, CoarseDirectRBSolver, 0, 1);
    FlexibleGeneralisedMinimalResidual<FermionField>    OuterSlv(1e-5, 200, VCycle, 2, true);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> OuterSchurRBSlv(OuterSlv);
    SchurSolverWrapper<FermionField>                    OuterSchurRBSolver(Dwc, OuterSchurRBSlv);
    OuterSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 2lvl vcycle as a preconditioner -- schur around outer solver");
  }
#endif

#if defined(THREE_LEVELS) && defined(SINGLE_APPLY)
  {
    printHeader("apply a single 3lvl vcycle -- everything on full grid");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSolver,
                                            CoarsePostSmoothSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, MdagMOp_Dc, CoarseVCycle, 0, 1);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 3lvl vcycle -- everything on full grid");
  }

  {
    printHeader("apply a single 3lvl vcycle -- schur smoothers and coarsest solver");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSchurRBSolver,
                                            CoarsePostSmoothSchurRBSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSchurRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSchurRBSolver, PostSmoothSchurRBSolver, MdagMOp_Dc, CoarseVCycle, 0, 1);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 3lvl vcycle -- schur smoothers and coarsest solver");
  }

  {
    printHeader("apply a single 3lvl vcycle -- schur around finest vcycle");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                          CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner                                VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseVCycle,
                                0,
                                1);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField> WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    WrappedVCycleSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 3.8e-1);
    printFooter("apply a single 3lvl vcycle -- schur around finest vcycle");
  }

  {
    printHeader("apply a single 3lvl vcycle -- everything in eo space");
    initializeFields(psi_o, r_o);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseVCycle,
                                0,
                                1);
    VCycle(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 3.8e-1);
    printFooter("apply a single 3lvl vcycle -- everything in eo space");
  }
#endif

#if defined(THREE_LEVELS) && defined(STANDALONE)
  {
    printHeader("use the 3lvl vcycle as a standalone solver -- everything on full grid");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSolver,
                                            CoarsePostSmoothSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, MdagMOp_Dc, CoarseVCycle, 0, 500, 5e-5);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 3lvl vcycle as a standalone solver -- everything on full grid");
  }

  {
    printHeader("use the 3lvl vcycle as a standalone solver -- schur smoothers and coarsest solver");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSchurRBSolver,
                                            CoarsePostSmoothSchurRBSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSchurRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(
      Aggs, MdagMOp_Dwc, PreSmoothSchurRBSolver, PostSmoothSchurRBSolver, MdagMOp_Dc, CoarseVCycle, 0, 500, 5e-5);
    VCycle(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 3lvl vcycle as a standalone solver -- schur smoothers and coarsest solver");
  }

  {
    printHeader("use the 3lvl vcycle as a standalone solver -- schur around finest vcycle");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                          CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner                                VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseVCycle,
                                0,
                                500, 5e-5);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField> WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    WrappedVCycleSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 5.0e-5);
    printFooter("use the 3lvl vcycle as a standalone solver -- schur around finest vcycle");
  }

  {
    printHeader("use the 3lvl vcycle as a standalone solver -- everything in eo space");
    initializeFields(psi_o, r_o);

    CoarseVCyclePreconditioner CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    VCyclePreconditioner       VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
                                CoarseVCycle,
                                0,
                                500,
                                5e-5);
    VCycle(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 5.0e-5);
    printFooter("use the 3lvl vcycle as a standalone solver -- everything in eo space");
  }
#endif

#if defined(THREE_LEVELS) && defined(PRECONDITIONER)
  {
    printHeader("use the 3lvl vcycle as a preconditioner -- everything on full grid");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                       CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSolver,
                                            CoarsePostSmoothSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSolver,
                                            1,
                                            1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(MdagMOp_Dc, CoarseOuterSlv);

    VCyclePreconditioner VCycle(Aggs,
                                MdagMOp_Dwc,
                                PreSmoothSolver,
                                PostSmoothSolver,
                                MdagMOp_Dc,
    #if defined(KCYCLE)
                                CoarseOuterSolver,
    #else
                                CoarseVCycle,
    #endif
                                0,
                                1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 3lvl vcycle as a preconditioner -- everything on full grid");
  }

  {
    printHeader("use the 3lvl vcycle as a preconditioner -- schur smoothers and coarsest solver");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                       CoarseVCycle(CoarseAggs,
                                            MdagMOp_Dc,
                                            CoarsePreSmoothSchurRBSolver,
                                            CoarsePostSmoothSchurRBSolver,
                                            MdagMOp_Dcc,
                                            CoarseCoarseSchurRBSolver,
                                            1,
                                            1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(MdagMOp_Dc, CoarseOuterSlv);

    VCyclePreconditioner VCycle(Aggs,
                                MdagMOp_Dwc,
                                PreSmoothSchurRBSolver,
                                PostSmoothSchurRBSolver,
                                MdagMOp_Dc,
#if defined(KCYCLE)
                                CoarseOuterSolver,
    #else
                                CoarseVCycle,
    #endif
                                0,
                                1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 3lvl vcycle as a preconditioner -- schur smoothers and coarsest solver");
  }

  {
    printHeader("use the 3lvl vcycle as a preconditioner -- schur around finest vcycle");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                       CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(DiagTwoOp_Dc, CoarseOuterSlv);
    VCyclePreconditioner                             VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
#if defined(KCYCLE)
                                CoarseOuterSolver,
    #else
                                CoarseVCycle,
    #endif
                                0,
                                1);
    OperatorFunctionWrapper<FermionField>               WrappedVCycle(VCycle);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> WrappedVCycleSchurRBSlv(WrappedVCycle);
    SchurSolverWrapper<FermionField>                 WrappedVCycleSchurRBSolver(Dwc, WrappedVCycleSchurRBSlv);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, WrappedVCycleSchurRBSolver, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);
    OuterSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 3lvl vcycle as a preconditioner -- schur around finest vcycle");
  }

  {
    printHeader("use the 3lvl vcycle as a preconditioner -- everything in eo space");
    initializeFields(psi_o, r_o);

    CoarseVCyclePreconditioner                       CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(DiagTwoOp_Dc, CoarseOuterSlv);
    VCyclePreconditioner                             VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
#if defined(KCYCLE)
                                CoarseOuterSolver,
    #else
                                CoarseVCycle,
    #endif
                                0,
                                1);
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(DiagTwoOp_Dwc, OuterSlv);
    OuterSolver(src_o, psi_o);

    assertCorrect(DiagTwoOp_Dwc, psi_o, r_o, src_o, 1.0e-5);
    printFooter("use the 3lvl vcycle as a preconditioner -- everything in eo space");
  }

  {
    printHeader("use the 3lvl vcycle as a preconditioner -- schur around outer solver");
    initializeFields(psi, r);

    CoarseVCyclePreconditioner                       CoarseVCycle(CoarseRBAggs,
                                            DiagTwoOp_Dc,
                                            CoarsePreSmoothDirectRBSolver,
                                            CoarsePostSmoothDirectRBSolver,
                                            DiagTwoOp_Dcc,
                                            CoarseCoarseDirectRBSolver,
                                            1,
                                            1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(DiagTwoOp_Dc, CoarseOuterSlv);
    VCyclePreconditioner                             VCycle(RBAggs,
                                DiagTwoOp_Dwc,
                                PreSmoothDirectRBSolver,
                                PostSmoothDirectRBSolver,
                                DiagTwoOp_Dc,
#if defined(KCYCLE)
                                CoarseOuterSolver,
    #else
                                CoarseVCycle,
    #endif
                                0,
                                1);
    FlexibleGeneralisedMinimalResidual<FermionField>    OuterSlv(1e-5, 200, VCycle, 2, true);
    NonHermitianSchurRedBlackDiagTwoSolve<FermionField> OuterSchurRBSlv(OuterSlv);
    SchurSolverWrapper<FermionField>                    OuterSchurRBSolver(Dwc, OuterSchurRBSlv);
    OuterSchurRBSolver(src, psi);

    assertCorrect(MdagMOp_Dwc, psi, r, src, 1.0e-5);
    printFooter("use the 3lvl vcycle as a preconditioner -- schur around outer solver");
  }
#endif

  Grid_finalize();
}


int main(int argc, char** argv) {
#if defined(NBASIS)
  runTest<NBASIS, NBASIS>(&argc, &argv);
#else
  int nbasis = readFromCommandLineInt(&argc, &argv, "--nbasis", 32);
  int nbasisc = readFromCommandLineInt(&argc, &argv, "--nbasisc", 32);

  assert((nbasis & 0x1) == 0 && "must be even");
  assert((nbasisc & 0x1) == 0 && "must be even");
  assert(nbasisc >= nbasis && "coarse basis size must be >= fine basis size");

  if      (nbasis == 12 && nbasisc == 12) runTest<12, 12>(&argc, &argv);
  else if (nbasis == 12 && nbasisc == 16) runTest<12, 16>(&argc, &argv);
  else if (nbasis == 12 && nbasisc == 24) runTest<12, 24>(&argc, &argv);
  else if (nbasis == 12 && nbasisc == 32) runTest<12, 32>(&argc, &argv);
  else if (nbasis == 12 && nbasisc == 48) runTest<12, 48>(&argc, &argv);
  else if (nbasis == 12 && nbasisc == 64) runTest<12, 64>(&argc, &argv);
  else if (nbasis == 16 && nbasisc == 16) runTest<16, 16>(&argc, &argv);
  else if (nbasis == 16 && nbasisc == 24) runTest<16, 24>(&argc, &argv);
  else if (nbasis == 16 && nbasisc == 32) runTest<16, 32>(&argc, &argv);
  else if (nbasis == 16 && nbasisc == 48) runTest<16, 48>(&argc, &argv);
  else if (nbasis == 16 && nbasisc == 64) runTest<16, 64>(&argc, &argv);
  else if (nbasis == 24 && nbasisc == 24) runTest<24, 24>(&argc, &argv);
  else if (nbasis == 24 && nbasisc == 32) runTest<24, 32>(&argc, &argv);
  else if (nbasis == 24 && nbasisc == 48) runTest<24, 48>(&argc, &argv);
  else if (nbasis == 24 && nbasisc == 64) runTest<24, 64>(&argc, &argv);
  else if (nbasis == 32 && nbasisc == 32) runTest<32, 32>(&argc, &argv);
  else if (nbasis == 32 && nbasisc == 48) runTest<32, 48>(&argc, &argv);
  else if (nbasis == 32 && nbasisc == 64) runTest<32, 64>(&argc, &argv);
  else if (nbasis == 48 && nbasisc == 48) runTest<48, 48>(&argc, &argv);
  else if (nbasis == 48 && nbasisc == 64) runTest<48, 64>(&argc, &argv);
  else if (nbasis == 64 && nbasisc == 64) runTest<64, 64>(&argc, &argv);
  else abort();
#endif
}
