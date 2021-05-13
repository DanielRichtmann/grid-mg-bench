/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/NullVectorSetupRework.h

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
    else Grid::BenchmarkHelpers::undoChiralDoublingG5C(Aggs.subspace);

    for(int n=0; n<nb; n++) FineSolver(null, Aggs.subspace[n]);

    basisOrthonormalize(Aggs.subspace, true); // gobal orthonormalization

    Grid::BenchmarkHelpers::performChiralDoublingG5C(Aggs.subspace);

    // block-wise orthonormalization (2 passes)
    typename Aggregates::CoarseScalar ip(Aggs.CoarseGrid);
    blockOrthogonalise(ip, Aggs.subspace);
    blockOrthogonalise(ip, Aggs.subspace);
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
