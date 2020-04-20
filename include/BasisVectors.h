/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/BasisVectors.h

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
#pragma once

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(Rework);

// clang-format off
struct MGBasisVectorsParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MGBasisVectorsParams,
                                  bool, preOrthonormalise,
                                  bool, postOrthonormalise,
                                  bool, testVectorSetup,
                                  bool, useHermOp,
                                  int,  maxIter
                                  );
};
// clang-format on

template<class Field>
class MGBasisVectors : public Profileable {
private:
  GridBase*          grid_;
  int                cb_;
  int                n_;
  std::vector<Field> V_;

public:
  MGBasisVectors(GridBase* grid, int cb, int n)
    : grid_(grid)
    , cb_(cb)
    , n_(n)
    , V_(n_, grid_)
  {}

  GridBase* Grid() const { return grid_; }
  int       Checkerboard() const { return cb_; }

  std::vector<Field> const& operator()() const { return V_; }
  std::vector<Field>&       operator()() { return V_; }

  void InitRandom(GridParallelRNG& rng) {
    prof_.Start("InitRandom.Total");
    for(auto& elem : V_) gaussian(rng, elem);
    prof_.Stop("InitRandom.Total");
  }

  void Generate(OperatorFunction<Field>&   solver,
                LinearOperatorBase<Field>& linop,
                MGBasisVectorsParams       params,
                std::function<void(int)>   innerLoopCallback = {},
                std::function<void()>      outerLoopCallback = {}) {
    prof_.Start("Generate.Total");
    prof_.Start("Generate.Misc");
    Field src(grid_);
    Field psi(grid_);
    Field Mpsi(grid_);
    prof_.Stop("Generate.Misc");

    for(int i = 0; i < params.maxIter; ++i) {
      if(params.preOrthonormalise) Orthonormalise();

      for(int k = 0; k < n_; ++k) {
        prof_.Start("Generate.ApplyOpAndNorm");
        if(params.useHermOp)
          linop.HermOp(V_[k], Mpsi);
        else
          linop.Op(V_[k], Mpsi);
        auto normBefore = norm2(Mpsi);
        prof_.Stop("Generate.ApplyOpAndNorm");

        prof_.Start("Generate.InitialValues");
        if(params.testVectorSetup) { // test vector setup (Lüscher, Wuppertal)
          psi = Zero();
          src = V_[k];
        } else { // null vector setup (QUDA)
          psi = V_[k];
          src = Zero();
        }
        prof_.Stop("Generate.InitialValues");

        prof_.Start("Generate.Misc");
        auto normInitGuess = norm2(psi);
        auto normSrc       = norm2(src);
        prof_.Stop("Generate.Misc");

        prof_.Start("Generate.ApplySolver");
        solver(linop, src, psi);
        V_[k] = psi;
        prof_.Stop("Generate.ApplySolver");

        prof_.Start("Generate.InnerLoopCallback");
        innerLoopCallback(k);
        prof_.Stop("Generate.InnerLoopCallback");

        prof_.Start("Generate.ApplyOpAndNorm");
        if(params.useHermOp)
          linop.HermOp(V_[k], Mpsi);
        else
          linop.Op(V_[k], Mpsi);
        auto normAfter = norm2(Mpsi);
        prof_.Stop("Generate.ApplyOpAndNorm");

        // clang-format off
        std::cout << GridLogMessage
                  << "Setup iter "            << i + 1
                  << " Vector "               << k
                  << " Initial guess = "      << normInitGuess
                  << " Initial rhs = "        << normSrc
                  << " <n|MdagM|n> before = " << normBefore
                  << " <n|MdagM|n> after = "  << normAfter
                  << " reduction = "          << normAfter / normBefore << std::endl;
        // clang-format on
      }

      if(params.postOrthonormalise) Orthonormalise();

      prof_.Start("Generate.OuterLoopCallback");
      outerLoopCallback();
      prof_.Stop("Generate.OuterLoopCallback");
    }
    prof_.Stop("Generate.Total");
  }

private:
  void Orthonormalise() {
    prof_.Start("Generate.Orthonormalise");
    orthogonalise(V_);
    prof_.Stop("Generate.Orthonormalise");
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
