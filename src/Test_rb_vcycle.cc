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


using namespace Grid;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::Rework;


#ifndef NBASIS
#define NBASIS 12
#endif


template<class FineObject, class CComplex, int nbasis, class CoarseSolver>
class NonHermitianVCyclePreconditioner : public LinearFunction<Lattice<FineObject>> {
public: // type definitions
  typedef Aggregation<FineObject, CComplex, nbasis>     Aggregates;
  typedef typename Aggregates::FineField                FineField;
  typedef LinearOperatorBase<FineField>                 FineOperator;
  typedef LinearFunction<FineField>                     FineSmoother;
  typedef CoarsenedMatrix<FineObject, CComplex, nbasis> CoarseOperator;
  typedef typename CoarseOperator::CoarseVector         CoarseField;

public: // sanity checks
  static_assert(nbasis % 2 == 0, "Must be even");
  static_assert(std::is_same<FineField, typename CoarseOperator::FineField>::value, "Type mismatch");

public: // data members
  Aggregates&      Aggregates_;
  FineOperator&    FineOperator_;
  FineSmoother&    PreSmoother_;
  FineSmoother&    PostSmoother_;
  CoarseOperator&  CoarseOperator_;
  CoarseSolver&    CoarseSolver_;
  int              level;
  int              maxIter;
  RealD            tolerance;
  bool             verboseConvergence;
  bool             verboseTiming;
  static const int nb = nbasis / 2;

public: // constructors
  NonHermitianVCyclePreconditioner(Aggregates&     Agg,
                                   FineOperator&   FineOp,
                                   FineSmoother&   PreSmooth,
                                   FineSmoother&   PostSmooth,
                                   CoarseOperator& CoarseOp,
                                   CoarseSolver&   CoarseSolve,
                                   int             lvl,
                                   int             maxIter,
                                   RealD           tolerance=0.0)
    : Aggregates_(Agg)
    , FineOperator_(FineOp)
    , PreSmoother_(PreSmooth)
    , PostSmoother_(PostSmooth)
    , CoarseOperator_(CoarseOp)
    , CoarseSolver_(CoarseSolve)
    , level(lvl)
    , maxIter(maxIter)
    , tolerance(tolerance)
    , verboseConvergence(true)
    , verboseTiming(true) {}

public: // member functions
  void operator()(FineField const& in, FineField& out) {
    // fields used in iteration
    FineField delta(in.Grid());
    FineField tmp(in.Grid());
    FineField r(in.Grid());
    CoarseField r_coarse(CoarseOperator_.Grid());
    CoarseField e_coarse(CoarseOperator_.Grid());

    // initial values, start with zero initial guess
    out = Zero(); r = in;

    // initial norms and residual
    double r2in = norm2(r);
    double r2   = r2in;
    double rsq  = tolerance * tolerance * r2in;

    // check for early convergence
    if (r2 <= rsq && tolerance != 0.0) return;

    for(int n=0; n<maxIter; n++) {
      // pre-smooth
      delta = Zero(); PreSmoother_(r, delta);

      // update solution
      add(out, out, delta);

      // update residuum, r -= M * delta
      FineOperator_.Op(delta, tmp); sub(r, r, tmp);

      // update residual
      double r2pre = (verboseConvergence) ? norm2(r) : 0.0; r2 = r2pre;

      // promote residual to coarse grid
      Aggregates_.ProjectToSubspace(r_coarse, r);

      // solve on coarse grid
      e_coarse = Zero(); CoarseSolver_(r_coarse, e_coarse);

      // promote coarse grid correction
      Aggregates_.PromoteFromSubspace(e_coarse, delta);

      // update solution, out += delta
      add(out, out, delta);

      // update residuum, r -= M * delta
      FineOperator_.Op(delta, tmp); sub(r, r, tmp);

      // update residual
      double r2coarse = (verboseConvergence) ? norm2(r) : 0.0; r2 = r2coarse;

      // post-smooth
      delta = Zero(); PostSmoother_(r, delta);

      // update solution, out += delta
      add(out, out, delta);

      // update residuum, r -= M * delta
      FineOperator_.Op(delta, tmp); sub(r, r, tmp);

      // update residual
      double r2post = norm2(r); r2 = r2post;

      // log
      if(verboseConvergence) {
        std::cout << GridLogMessage << "NonHermitianVCycle: level = " << level << " cycle = " << n
                  << " relative squared residual norms:"
                  << " initially "      << r2in/r2in
                  << " pre-smoothing "  << r2pre/r2in
                  << " coarse-solve "   << r2coarse/r2in
                  << " post-smoothing " << r2post/r2in
                  << std::endl;
      }

      // terminate when target reached
      if(r2 <= rsq) {
        if(tolerance != 0.0) {
          FineOperator_.Op(out, r); sub(r, r, in);

          RealD srcnorm       = sqrt(r2in);
          RealD resnorm       = sqrt(norm2(r));
          RealD true_residual = resnorm / srcnorm;

          std::cout << GridLogMessage        << "NonHermitianVCycle: Converged on iteration " << n+1
                    << " computed residual " << sqrt(r2/r2in)
                    << " true residual "     << true_residual
                    << " target "            << tolerance << std::endl;
        }
        return;
      }
    }
  }
};


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
    FineField null(Aggs.FineGrid); null = Zero();
    const int nb = nbasis/2;

    if(fromRandom) for(int n=0; n<nb; n++) gaussian(RNG_, Aggs.subspace[n]);
    else undoChiralDoublingG5C(Aggs.subspace);

    for(int n=0; n<nb; n++) FineSolver(null, Aggs.subspace[n]); // TODO: modify solvers to work with this!

    basisOrthonormalize(Aggs.subspace, true); // gobal orthonormalization

    performChiralDoublingG5C(Aggs.subspace);

    // block-wise orthonormalization (2 passes)
    typename Aggregates::CoarseScalar ip(Aggs.CoarseGrid);
    blockOrthogonalise(ip, Aggs.subspace);
    blockOrthogonalise(ip, Aggs.subspace);
  }
};


int main(int argc, char** argv) {
  Grid_init(&argc, &argv);
  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  const int nbasis = NBASIS; static_assert((nbasis & 0x1) == 0, "");
  const int  nb        = nbasis / 2;
  Coordinate blockSize = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({2, 2, 2, 2}));

  std::cout << GridLogMessage << "Compiled with nbasis = " << nbasis << " -> nb = " << nb << std::endl;

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
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

  typedef Aggregation<vSpinColourVector, vTComplex, nbasis>                 Aggregates;
  typedef CoarsenedMatrix<vSpinColourVector, vTComplex, nbasis>             CoarseOperator;
  typedef CoarseOperator::CoarseVector                                      CoarseVector;
  typedef CoarseOperator::siteVector                                        CoarseSiteVector;
  typedef NonHermitianNullVectorSetup<vSpinColourVector, vTComplex, nbasis> NullVectorSetup;
  typedef NonHermitianVCyclePreconditioner<vSpinColourVector, vTComplex, nbasis, LinearFunction<CoarseVector>>
                                             VCyclePreconditioner;
  typedef WilsonCloverFermionR               WilsonCloverOperator;
  typedef WilsonCloverFermionR::FermionField FermionField;

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG, Umu);

  RealD mass = 0.10; RealD csw = 1.25;

  WilsonCloverOperator                                    Dwc(Umu, *FGrid, *FrbGrid, mass, csw, csw);
  MdagMLinearOperator<WilsonCloverOperator, FermionField> MdagMOp_Dwc(Dwc);

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  Aggregates                                          Aggs(CGrid, FGrid, 0);
  BiCGSTAB<FermionField>                              SetupSlv(1e-5, 500, false);
  // GeneralisedMinimalResidual<FermionField>            SetupSlv(1e-5, 500, 10, false);
  SolverWrapper<FermionField>                         SetupSolver(MdagMOp_Dwc, SetupSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> SetupRBSlv(SetupSlv);
  SchurSolverWrapper<FermionField>                    SetupRBSolver(Dwc, SetupRBSlv);
  NullVectorSetup                                     Setup(MdagMOp_Dwc, FPRNG);
  Setup(Aggs, SetupSolver);
  // Setup(Aggs, SetupRBSolver); // this doesn't work just yet :/

  /////////////////////////////////////////////////////////////////////////////
  //                  Setup of CoarsenedMatrix and Operator                  //
  /////////////////////////////////////////////////////////////////////////////

  const int hermitian = 0;
  CoarseOperator Dc(*CGrid, *CrbGrid, hermitian); Dc.CoarsenOperator(FGrid, MdagMOp_Dwc, Aggs);
  MdagMLinearOperator<CoarseOperator, CoarseVector> MdagMOp_Dc(Dc);

  /////////////////////////////////////////////////////////////////////////////
  //                      Setup of VCycle preconditioner                     //
  /////////////////////////////////////////////////////////////////////////////

  // unpreconditioned solvers + their wrappers
  MinimalResidual<FermionField>                       PreSmoothSlv(0.1, 4, 1.1, false);
  MinimalResidual<FermionField>                       PostSmoothSlv(0.1, 4, 1.1, false);
  GeneralisedMinimalResidual<CoarseVector>            CoarseSlv(0.1, 200, 10, false);
  SolverWrapper<FermionField>                         PreSmoothSolver(MdagMOp_Dwc, PreSmoothSlv);
  SolverWrapper<FermionField>                         PostSmoothSolver(MdagMOp_Dwc, PostSmoothSlv);
  SolverWrapper<CoarseVector>                         CoarseSolver(MdagMOp_Dc, CoarseSlv);

  // preconditioned solvers + their wrappers
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> PreSmoothRBSlv(PreSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<FermionField> PostSmoothRBSlv(PostSmoothSlv);
  NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector> CoarseRBSlv(CoarseSlv);
  SchurSolverWrapper<FermionField>                    PreSmoothRBSolver(Dwc, PreSmoothRBSlv);
  SchurSolverWrapper<FermionField>                    PostSmoothRBSolver(Dwc, PostSmoothRBSlv);
  SchurSolverWrapper<CoarseVector>                    CoarseRBSolver(Dc, CoarseRBSlv);

  /////////////////////////////////////////////////////////////////////////////
  //                                  Tests                                  //
  /////////////////////////////////////////////////////////////////////////////

#define USE_EO

  {
    LatticeFermion src(FGrid); gaussian(FPRNG, src);
    LatticeFermion psi(FGrid); psi = Zero();
    LatticeFermion r(FGrid);   r = Zero();

#if defined(USE_EO)
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothRBSolver, PostSmoothRBSolver, Dc, CoarseRBSolver, 0, 1); // eo in smoothers + coarse grid solve
#else
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothSolver,   PostSmoothSolver,   Dc, CoarseSolver,   0, 1); // non-eo
#endif
    VCycle(src, psi);

    MdagMOp_Dwc.Op(psi, r); sub(r, r, src);
    assert(sqrt(norm2(r)/norm2(src)) < 3.8e-1);
    std::cout << GridLogMessage << "****** Test \"apply a single vcycle\" passed" << std::endl;
  }

  {
    LatticeFermion src(FGrid); gaussian(FPRNG, src);
    LatticeFermion psi(FGrid); psi = Zero();
    LatticeFermion r(FGrid);   r = Zero();

#if defined(USE_EO)
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothRBSolver, PostSmoothRBSolver, Dc, CoarseRBSolver, 0, 500, 5e-5); // eo in smoothers + coarse grid solve
#else
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothSolver,   PostSmoothSolver,   Dc, CoarseSolver,   0, 500, 5e-5); // non-eo
#endif
    VCycle(src, psi);

    MdagMOp_Dwc.Op(psi, r); sub(r, r, src);
    assert(sqrt(norm2(r) / norm2(src)) < 5.0e-5);
    std::cout << GridLogMessage << "****** Test \"use the vcycle as a standalone solver\" passed" << std::endl;
  }

  {
    LatticeFermion src(FGrid); gaussian(FPRNG, src);
    LatticeFermion psi(FGrid); psi = Zero();
    LatticeFermion r(FGrid);   r = Zero();

#if defined(USE_EO)
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothRBSolver, PostSmoothRBSolver, Dc, CoarseRBSolver, 0, 1); // eo in smoothers + coarse grid solve
#else
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothSolver,   PostSmoothSolver,   Dc, CoarseSolver,   0, 1); // non-eo
#endif
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);

    OuterSolver(src, psi);

    MdagMOp_Dwc.Op(psi, r); sub(r, r, src);
    assert(sqrt(norm2(r) / norm2(src)) < 1.0e-5);
    std::cout << GridLogMessage << "******* Test \"use the vcycle as a 2lvl preconditioner\" passed" << std::endl;
  }

  {
    LatticeFermion src(FGrid); gaussian(FPRNG, src);
    LatticeFermion psi(FGrid); psi = Zero();
    LatticeFermion r(FGrid);   r = Zero();

    // constants
    const int nbasisc = nbasis;
    const int nbc = nb;

    // type definitions
    typedef Aggregation<CoarseSiteVector, iScalar<vTComplex>, nbasisc>                 CoarseAggregates;
    typedef CoarsenedMatrix<CoarseSiteVector, iScalar<vTComplex>, nbasisc>             CoarseCoarseOperator;
    typedef CoarseCoarseOperator::CoarseVector                                         CoarseCoarseVector;
    typedef NonHermitianNullVectorSetup<CoarseSiteVector, iScalar<vTComplex>, nbasisc> CoarseNullVectorSetup;
    typedef NonHermitianVCyclePreconditioner<CoarseSiteVector,
                                             iScalar<vTComplex>,
                                             nbasisc,
                                             LinearFunction<CoarseCoarseVector>>
      CoarseVCyclePreconditioner;

    // general setup
    Coordinate cblockSize = readFromCommandLineCoordinate(&argc, &argv, "--cblocksize", Coordinate({2, 2, 2, 2}));
    Coordinate cclatt = calcCoarseLattSize(CGrid->_fdimensions, cblockSize);
    GridCartesian*         CCGrid   = SpaceTimeGrid::makeFourDimGrid(cclatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian* CCrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CCGrid);
    std::cout << GridLogMessage << "CCGrid:" << std::endl; CGrid->show_decomposition();
    std::cout << GridLogMessage << "CCrbGrid:" << std::endl; CrbGrid->show_decomposition();
    GridParallelRNG CCPRNG(CCGrid); CPRNG.SeedFixedIntegers(seeds);

    // setup of coarse Aggregation
    CoarseAggregates                         CoarseAggs(CCGrid, CGrid, 0);
    BiCGSTAB<CoarseVector>                   CoarseSetupSlv(1e-5, 500, false);
    // GeneralisedMinimalResidual<CoarseVector> CoarseSetupSlv(1e-5, 500, 10, false);
    SolverWrapper<CoarseVector>              CoarseSetupSolver(MdagMOp_Dc, CoarseSetupSlv);
    CoarseNullVectorSetup                    CoarseSetup(MdagMOp_Dc, CCPRNG);
    CoarseSetup(CoarseAggs, CoarseSetupSolver);

    // setup of coarse CoarsenedMatrix and Operator
    CoarseCoarseOperator Dcc(*CCGrid, *CCrbGrid, hermitian); Dcc.CoarsenOperator(CGrid, MdagMOp_Dc, CoarseAggs);
    MdagMLinearOperator<CoarseCoarseOperator, CoarseCoarseVector> MdagMOp_Dcc(Dcc);

    // setup of coarse VCycle preconditioner -- unpreconditioned solvers + their wrappers
    MinimalResidual<CoarseVector>                  CoarsePreSmoothSlv(0.1, 4, 1.1, false);
    MinimalResidual<CoarseVector>                  CoarsePostSmoothSlv(0.1, 4, 1.1, false);
    GeneralisedMinimalResidual<CoarseCoarseVector> CoarseCoarseSlv(0.1, 200, 10, false);
    SolverWrapper<CoarseVector>                    CoarsePreSmoothSolver(MdagMOp_Dc, CoarsePreSmoothSlv);
    SolverWrapper<CoarseVector>                    CoarsePostSmoothSolver(MdagMOp_Dc, CoarsePostSmoothSlv);
    SolverWrapper<CoarseCoarseVector>              CoarseCoarseSolver(MdagMOp_Dcc, CoarseCoarseSlv);

    // setup of coarse VCycle preconditioner -- preconditioned solvers + their wrappers
    NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector>       CoarsePreSmoothRBSlv(CoarsePreSmoothSlv);
    NonHermitianSchurRedBlackDiagTwoSolve<CoarseVector>       CoarsePostSmoothRBSlv(CoarsePostSmoothSlv);
    NonHermitianSchurRedBlackDiagTwoSolve<CoarseCoarseVector> CoarseCoarseRBSlv(CoarseCoarseSlv);
    SchurSolverWrapper<CoarseVector>                          CoarsePreSmoothRBSolver(Dc, CoarsePreSmoothRBSlv);
    SchurSolverWrapper<CoarseVector>                          CoarsePostSmoothRBSolver(Dc, CoarsePostSmoothRBSlv);
    SchurSolverWrapper<CoarseCoarseVector>                    CoarseCoarseRBSolver(Dcc, CoarseCoarseRBSlv);

    // setup of coarse outer solver (= wrapper around coarse vcycle) for kcycle
    CoarseVCyclePreconditioner CoarseVCycle(CoarseAggs, MdagMOp_Dc, CoarsePreSmoothSolver, CoarsePostSmoothSolver, Dcc, CoarseCoarseSolver, 1, 1);
    FlexibleGeneralisedMinimalResidual<CoarseVector> CoarseOuterSlv(0.1, 200, CoarseVCycle, 10, true);
    SolverWrapper<CoarseVector>                      CoarseOuterSolver(MdagMOp_Dc, CoarseOuterSlv);

    // setup of VCycle preconditioner, using either coarse VCycle (-> vcycle) or coarse outer solver as coarse solver (-> kcycle)
    // VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, Dc, CoarseVCycle, 0, 1);      // v-cycle
    VCyclePreconditioner VCycle(Aggs, MdagMOp_Dwc, PreSmoothSolver, PostSmoothSolver, Dc, CoarseOuterSolver, 0, 1); // k-cycle
    FlexibleGeneralisedMinimalResidual<FermionField> OuterSlv(1e-5, 200, VCycle, 2, true);
    SolverWrapper<FermionField>                      OuterSolver(MdagMOp_Dwc, OuterSlv);

    OuterSolver(src, psi);

    MdagMOp_Dwc.Op(psi, r); sub(r, r, src);
    assert(sqrt(norm2(r) / norm2(src)) < 1.0e-5);
    std::cout << GridLogMessage << "******* Test \"use the vcycle as a 3lvl preconditioner\" passed" << std::endl;
  }

  Grid_finalize();
}
