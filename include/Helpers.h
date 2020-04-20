/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/MultiGridPreconditioner.h

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
#pragma once

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(Rework);

#define LOG_MG_LVL std::cout << GridLogMG << "Level " << _Level << ": "
#define LVL_STR "MG_Level_" + std::to_string(_Level) + "::"

template<class Field>
std::unique_ptr<OperatorFunction<Field>> createUtilisedSolver(MGUtilizedSolverType const&          solverType,
                                                              MultiGridUtilizedSolverParams const& par,
                                                              LinearFunction<Field>&               prec) {
  if(solverType == MGUtilizedSolverType::MR) {
    return std::unique_ptr<OperatorFunction<Field>>(
      new MinimalResidual<Field>(par.tolerance, par.maxIter, 1.0, false, false));
  } else if(solverType == MGUtilizedSolverType::GMRES) {
    return std::unique_ptr<OperatorFunction<Field>>(
      new GeneralisedMinimalResidual<Field>(par.tolerance, par.maxIter, par.restartLength, false, false));
  } else if(solverType == MGUtilizedSolverType::FGMRES) {
    return std::unique_ptr<OperatorFunction<Field>>(new FlexibleGeneralisedMinimalResidual<Field>(
      par.tolerance, par.maxIter, prec, par.restartLength, false, false));
  } else if(solverType == MGUtilizedSolverType::BiCGSTAB) {
    return std::unique_ptr<OperatorFunction<Field>>(
      new BiConjugateGradientStabilised<Field>(par.tolerance, par.maxIter, false, false));
  } else
    assert(0);
}

template<class Field>
std::unique_ptr<SchurRedBlackBase<Field>> createUtilisedRBSolver(MGUtilizedSolverType const& solverType,
                                                                 OperatorFunction<Field>&    solver) {
  // TODO: This currently only works for non-herm solvers! Need a toggle between herm and non herm
  return std::unique_ptr<SchurRedBlackBase<Field>>(
    new SchurRedBlackDiagMooeeNonHermSolve<Field>(solver, false, true));
}

template<class Field>
void analyseTestVectors(LinearOperatorBase<Field>& Linop, std::vector<Field> const& vectors, int nn) {
  std::vector<Field> tmp(4, vectors[0].Grid());

  std::cout << GridLogMessage << "Test vector analysis:" << std::endl;

  auto positiveOnes = 0;
  for(auto i = 0; i < nn; ++i) {
    Linop.Op(vectors[i], tmp[3]);

    G5C(tmp[0], tmp[3]);

    auto lambda = innerProduct(vectors[i], tmp[0]) / innerProduct(vectors[i], vectors[i]);

    tmp[1] = tmp[0] - lambda * vectors[i];

    auto mu  = ::sqrt(norm2(tmp[1]) / norm2(vectors[i]));
    auto nrm = ::sqrt(norm2(vectors[i]));

    if(real(lambda) > 0) positiveOnes++;

    std::cout << GridLogMessage << std::scientific << std::setprecision(2) << std::setw(2) << std::showpos
              << "vector " << i << ": "
              << "singular value: " << lambda << ", singular vector precision: " << mu << ", norm: " << nrm
              << std::endl;
  }
  std::cout << GridLogMessage << std::scientific << std::setprecision(2) << std::setw(2) << std::showpos
            << positiveOnes << " out of " << nn << " vectors were positive" << std::endl;
  std::cout << std::noshowpos;
}

Coordinate calcCoarseLattSize(Coordinate const& fineLattSize, Coordinate const& blockSize) {
  Coordinate ret(fineLattSize);
  for(int d = 0; d < ret.size(); ++d) ret[d] /= blockSize[d];
  return ret;
}

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
