/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/VcycleRework.h

    Copyright (C) 2015 - 2021

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
#pragma once

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(Rework);

template<class FineObject, class CComplex, int nbasis, class CoarseSolver>
class NonHermitianVCyclePreconditioner : public LinearFunction<Lattice<FineObject>> {
public: // type definitions
  typedef Aggregation<FineObject, CComplex, nbasis>     Aggregates;
  typedef typename Aggregates::FineField                FineField;
  typedef LinearOperatorBase<FineField>                 FineOperator;
  typedef LinearFunction<FineField>                     FineSmoother;
  typedef typename Aggregates::CoarseVector             CoarseField;
  typedef LinearOperatorBase<CoarseField>               CoarseOperator;

public: // sanity checks
  static_assert(nbasis % 2 == 0, "Must be even");
  static_assert(std::is_same<FineField, Lattice<FineObject>>::value, "Type mismatch");

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
    , verboseConvergence(false)
    , verboseTiming(true) {}

public: // member functions
  void operator()(FineField const& in, FineField& out) {
    // grids
    GridBase* Grid_f = Aggregates_.FineGrid;
    GridBase* Grid_c = Aggregates_.CoarseGrid;
    conformable(Grid_f, in.Grid());
    conformable(Grid_f, out.Grid());
    conformable(in, out);

    // fields used in iteration
    FineField delta(Grid_f);
    FineField tmp(Grid_f);
    FineField r(Grid_f);
    CoarseField r_coarse(Grid_c);
    CoarseField e_coarse(Grid_c);

    // correct checkerboards
    out.Checkerboard()      = in.Checkerboard();
    delta.Checkerboard()    = in.Checkerboard();
    tmp.Checkerboard()      = in.Checkerboard();
    r.Checkerboard()        = in.Checkerboard();
    r_coarse.Checkerboard() = in.Checkerboard();
    e_coarse.Checkerboard() = in.Checkerboard();

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

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
