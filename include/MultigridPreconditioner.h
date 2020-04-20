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

template<class Field>
class MultiGridPreconditionerBase : public LinearFunction<Field> {
public:
  virtual ~MultiGridPreconditionerBase()                      = default;
  virtual void initialSetup()                                 = 0;
  virtual void refinementSetup()                              = 0;
  virtual void operator()(Field const& in, Field& out)        = 0;
  virtual void runChecks(RealD tolerance)                     = 0;
  virtual void reportTimings()                                = 0;
};

template<class Fobj, class CComplex, int nBasis, int nCoarseSpins, int nCoarserLevels, class Matrix>
class MultiGridPreconditioner : public MultiGridPreconditionerBase<Lattice<Fobj>> {
private:

  /////////////////////////////////////////////
  // Private implementations
  /////////////////////////////////////////////

  template<class Field>
  class OperatorFunctionWrapper : public OperatorFunction<Field> {
  public:
    OperatorFunctionWrapper(LinearFunction<Field>& fn) : fn_(fn){};
    virtual ~OperatorFunctionWrapper(void) = default;
    virtual void operator()(
      LinearOperatorBase<Field>& op,
      const Field&               in,
      Field&                     out) {
      fn_(in, out);
    }

  private:
    LinearFunction<Field>& fn_;
  };

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

public:
  static_assert(nCoarseSpins == 1 || nCoarseSpins == 2 || nCoarseSpins == 4,
                "Number of coarse spins invalid");

  // clang-format off
#if defined(USE_NEW_COARSENING)
  typedef CoarseningPolicy<Lattice<Fobj>, nBasis, nCoarseSpins>                                                            UtilisedCoarseningPolicy;
  typedef Aggregation<UtilisedCoarseningPolicy>                                                                            Aggregates;
  typedef CoarsenedMatrix<UtilisedCoarseningPolicy>                                                                        CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField                                                                         CoarseVector;
  typedef typename CoarseDiracMatrix::SiteSpinor                                                                           CoarseSiteVector;
  typedef Matrix                                                                                                           FineDiracMatrix;
  typedef typename CoarseDiracMatrix::FineFermionField                                                                     FineVector;
  typedef MGBasisVectors<FineVector>                                                                                       BasisVectors;
  typedef MultiGridPreconditioner<CoarseSiteVector, CComplex, nBasis, nCoarseSpins, nCoarserLevels - 1, CoarseDiracMatrix> NextPreconditionerLevel;
#else
  typedef Grid::Aggregation<Fobj, CComplex, nBasis>                                                                                 Aggregates;
  typedef Grid::CoarsenedMatrix<Fobj, CComplex, nBasis>                                                                             CoarseDiracMatrix;
  typedef typename Aggregates::CoarseVector                                                                                         CoarseVector;
  typedef typename Aggregates::siteVector                                                                                           CoarseSiteVector;
  typedef Matrix                                                                                                                    FineDiracMatrix;
  typedef typename Aggregates::FineField                                                                                            FineVector;
  typedef MGBasisVectors<FineVector>                                                                                                BasisVectors;
  typedef MultiGridPreconditioner<CoarseSiteVector, iScalar<CComplex>, nBasis, nCoarseSpins, nCoarserLevels - 1, CoarseDiracMatrix> NextPreconditionerLevel;
#endif
  // clang-format on

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

#if defined(USE_NEW_COARSENING)
  const int nB = nBasis;
#else
  static_assert((nBasis & 0x1) == 0, "MG Preconditioner only supports an even number of basis vectors");
  const int nB = nBasis / 2;
#endif

  int _Level;
  int _NextLevel;

  MultiGridParams& _Params;
  LevelInfo&       _LevelInfo;

  FineDiracMatrix& _FineMatrix;
  FineDiracMatrix& _SmootherMatrix;

  BasisVectors      _BasisVectors;
  Aggregates        _Aggregates;
  CoarseDiracMatrix _CoarseMatrix;

  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineMdagMOp;
  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineSmootherMdagMOp;

  FineVector _FineSrc;
  FineVector _FineSol;

  Profiler _Prof;

  std::unique_ptr<NextPreconditionerLevel> _NextPreconditionerLevel;

  /////////////////////////////////////////////
  // Member Functions (Interface)
  /////////////////////////////////////////////

  MultiGridPreconditioner(
    MultiGridParams& Params,
    LevelInfo&       LvlInfo,
    FineDiracMatrix& FineMat,
    FineDiracMatrix& SmootherMat)
    : _Level(Params.nLevels - (nCoarserLevels + 1)) // _Level = 0 corresponds to finest
    , _NextLevel(_Level + 1)                   // incremented for instances on coarser levels
    , _Params(Params)
    , _LevelInfo(LvlInfo)
    , _FineMatrix(FineMat)
    , _SmootherMatrix(SmootherMat)
    , _BasisVectors(_LevelInfo.Grids[_Level], 0, nB)
#if defined(USE_NEW_COARSENING)
    , _Aggregates(_LevelInfo.Grids[_NextLevel], _LevelInfo.Grids[_Level], 0, 1)        // use faster versions
    , _CoarseMatrix(*_LevelInfo.Grids[_NextLevel], *_LevelInfo.RBGrids[_NextLevel], 1) // of both
#else
    , _Aggregates(_LevelInfo.Grids[_NextLevel], _LevelInfo.Grids[_Level], 0)
    , _CoarseMatrix(*_LevelInfo.Grids[_NextLevel], *_LevelInfo.RBGrids[_NextLevel])
#endif
    , _FineMdagMOp(_FineMatrix)
    , _FineSmootherMdagMOp(_SmootherMatrix)
    , _FineSrc(_LevelInfo.Grids[_Level])
    , _FineSol(_LevelInfo.Grids[_Level]) {

    _NextPreconditionerLevel = std::unique_ptr<NextPreconditionerLevel>(
      new NextPreconditionerLevel(_Params, _LevelInfo, _CoarseMatrix, _CoarseMatrix));
  }

  void initialSetup() {
    if(_Level == 0) _Prof.Start("AllLevels.Global");
    _Prof.Start("ThisLevel.Total");

    LOG_MG_LVL << "Running initial setup phase" << std::endl;

    if(_Level == 0 || _Params.setupGenerateOnAllLevels) {
      LOG_MG_LVL << "Generating basis vectors from random" << std::endl;

      _BasisVectors.InitRandom(_LevelInfo.PRNGs[_Level]);

      _Prof.Start("ThisLevel.Misc");
      TrivialPrecon<FineVector> simplePrec;
      auto                      solver = createUtilisedSolver(
        _Params.setupSolverType[_Level],
        _Params.setupSolverParams[_Level],
        simplePrec);

      MGBasisVectorsParams bvParams;
      bvParams.preOrthonormalise  = _Params.setupPreOrthonormalise;
      bvParams.postOrthonormalise = _Params.setupPostOrthonormalise;
      bvParams.testVectorSetup    = (_Params.setupBasisVectorsType == MGBasisVectorsType::testVectors);
      bvParams.maxIter            = _Params.setupInitialIter[_Level];
      _Prof.Stop("ThisLevel.Misc");

      _Prof.Start("ThisLevel.GenerateBasisVecs");
      _BasisVectors.Generate(
        *solver,
        _FineSmootherMdagMOp,
        bvParams,
        [](int n) {},
        []() {});
      _Prof.Stop("ThisLevel.GenerateBasisVecs");
    } else
      LOG_MG_LVL << "Re-using projected basis vectors from previous level"
                 << std::endl;

    recreateOperators(false);

    if(!_Params.setupGenerateOnAllLevels) projectBasisVectorsDownwardsIfNecessary();

    _Prof.Stop("ThisLevel.Total");
    _NextPreconditionerLevel->initialSetup();

    if(_Level == 0) _Prof.Stop("AllLevels.Global");
    if(_Level == 0) printProfiling(LVL_STR + "AllLevels.Global", "initSetup");
  }

  void refinementSetup() {
    if(_Params.setupRefinementIter[_Level] > 0) {
      if(_Level == 0) _Prof.Start("AllLevels.Global");
      _Prof.Start("ThisLevel.Total");

      _Prof.Start("ThisLevel.Misc");
      LOG_MG_LVL << "Running new debugging setup phase with " << _Params.setupRefinementIter[_Level]
                 << " iterations" << std::endl;

      OperatorFunctionWrapper<FineVector> me(*this);

      MGBasisVectorsParams bvParams;
      bvParams.preOrthonormalise  = _Params.setupPreOrthonormalise;
      bvParams.postOrthonormalise = _Params.setupPostOrthonormalise;
      bvParams.testVectorSetup    = true; // null-vector setup doesn't work in iterative refinement setup phase
      bvParams.maxIter            = _Params.setupRefinementIter[_Level];
      _Prof.Stop("ThisLevel.Misc");

      _Prof.Start("ThisLevel.GenerateBasisVecs");
      _BasisVectors.Generate(
        // *solver, // quda approach = krylov solver with MG prec
        me, // wuppertal approach = only MG prec
        _FineSmootherMdagMOp,
        bvParams,
        [this](int n) {
          updateBasisVector(_Params.setupRecursiveUpdates, _Level, n);
        },
        [this]() {
          recreateOperators(_Params.setupRecursiveUpdates);
        });
      _Prof.Stop("ThisLevel.GenerateBasisVecs");

      _Prof.Stop("ThisLevel.Total");
      _NextPreconditionerLevel->refinementSetup();

      if(_Level == 0) _Prof.Stop("AllLevels.Global");
      if(_Level == 0) printProfiling(LVL_STR + "AllLevels.Global", "iterSetup");
    }
  }

  virtual void operator()(FineVector const& in, FineVector& out) {
    conformable(_LevelInfo.Grids[_Level], in.Grid());
    conformable(in, out);

    if(_Level == 0) _Prof.Start("AllLevels.Op.Global");
    _Prof.Start("ThisLevel.Op.Total");
    _Prof.Start("ThisLevel.Op.Misc");
    _NextPreconditionerLevel->_FineSol = Zero();

    TrivialPrecon<FineVector> simplePrec;

    auto smootherSolver = createUtilisedSolver(
      _Params.smootherSolverType[_Level],
      _Params.smootherSolverParams[_Level],
      simplePrec);
    auto wrapperSolver = createUtilisedSolver(
      _Params.wrapperSolverType[_Level],
      _Params.wrapperSolverParams[_Level],
      *_NextPreconditionerLevel);

    auto smootherRBSolver = createUtilisedRBSolver(_Params.smootherSolverType[_Level], *smootherSolver);
    auto wrapperRBSolver  = createUtilisedRBSolver(_Params.wrapperSolverType[_Level], *wrapperSolver);
    _Prof.Stop("ThisLevel.Op.Misc");

    _Prof.Start("ThisLevel.Op.Restrict");
    _Aggregates.ProjectToSubspace(_NextPreconditionerLevel->_FineSrc, in);
    _Prof.Stop("ThisLevel.Op.Restrict");

    _Prof.Stop("ThisLevel.Op.Total");
    if(_Params.cyclingStrategy[_Level] == MGCyclingStrategy::KCYCLE) {
      _Prof.Start("ThisLevel.Op.Wrapper");
      (*wrapperSolver)(
        _NextPreconditionerLevel->_FineMdagMOp, _NextPreconditionerLevel->_FineSrc, _NextPreconditionerLevel->_FineSol);
      _Prof.Stop("ThisLevel.Op.Wrapper");
    } else if(_Params.cyclingStrategy[_Level] == MGCyclingStrategy::VCYCLE)
      (*_NextPreconditionerLevel)(_NextPreconditionerLevel->_FineSrc, _NextPreconditionerLevel->_FineSol);
    _Prof.Start("ThisLevel.Op.Total");

    _Prof.Start("ThisLevel.Op.Prolongate");
    _Aggregates.PromoteFromSubspace(_NextPreconditionerLevel->_FineSol, out);
    _Prof.Stop("ThisLevel.Op.Prolongate");

    RealD residualAfterCoarseGridCorrection = calcResidual(_FineMdagMOp, in, out);

    _Prof.Start("ThisLevel.Op.Smoother");
    if(_Params.smootherSolverParams[_Level].useRBPrec)
      (*smootherRBSolver)(_SmootherMatrix, in, out);
    else
      (*smootherSolver)(_FineSmootherMdagMOp, in, out);
    _Prof.Stop("ThisLevel.Op.Smoother");

    RealD residualAfterPostSmoother = calcResidual(_FineMdagMOp, in, out);
    printInfo(norm2(in), residualAfterCoarseGridCorrection, residualAfterPostSmoother);

    _Prof.Stop("ThisLevel.Op.Total");
    if(_Level == 0) _Prof.Stop("AllLevels.Op.Global");
  }

  void runChecks(RealD tolerance) {
    LOG_MG_LVL << "Running MG correctness checks" << std::endl;

    std::vector<FineVector>   fineTmps(7, _LevelInfo.Grids[_Level]);
    std::vector<CoarseVector> coarseTmps(4, _LevelInfo.Grids[_NextLevel]);

    LOG_MG_LVL << "**************************************************" << std::endl;
    LOG_MG_LVL << "MG correctness check: 0 == (M - (Mdiag + Σ_μ Mdir_μ)) * v" << std::endl;
    LOG_MG_LVL << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_Level], fineTmps[0]);

    _FineMdagMOp.Op(fineTmps[0], fineTmps[1]);     //     M * v
    _FineMdagMOp.OpDiag(fineTmps[0], fineTmps[2]); // Mdiag * v

    fineTmps[4] = Zero();
    for(int dir = 0; dir < 4; dir++) { //       Σ_μ Mdir_μ * v
      for(auto disp : {+1, -1}) {
        _FineMdagMOp.OpDir(fineTmps[0], fineTmps[3], dir, disp);
        fineTmps[4] = fineTmps[4] + fineTmps[3];
      }
    }

    fineTmps[5] = fineTmps[2] + fineTmps[4]; // (Mdiag + Σ_μ Mdir_μ) * v

    fineTmps[6]    = fineTmps[1] - fineTmps[5];
    auto deviation = std::sqrt(norm2(fineTmps[6]) / norm2(fineTmps[1]));

    LOG_MG_LVL << "norm2(M * v)                    = " << norm2(fineTmps[1]) << std::endl;
    LOG_MG_LVL << "norm2(Mdiag * v)                = " << norm2(fineTmps[2]) << std::endl;
    LOG_MG_LVL << "norm2(Σ_μ Mdir_μ * v)           = " << norm2(fineTmps[4]) << std::endl;
    LOG_MG_LVL << "norm2((Mdiag + Σ_μ Mdir_μ) * v) = " << norm2(fineTmps[5]) << std::endl;
    LOG_MG_LVL << "relative deviation              = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    LOG_MG_LVL << "**************************************************" << std::endl;
    LOG_MG_LVL << "MG correctness check: 0 == (1 - P R) v" << std::endl;
    LOG_MG_LVL << "**************************************************" << std::endl;

    for(auto i = 0; i < _Aggregates.Subspace().size(); ++i) {
      _Aggregates.ProjectToSubspace(coarseTmps[0], _Aggregates.Subspace()[i]); //   R v_i
      _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]);             // P R v_i

      fineTmps[1] = _Aggregates.Subspace()[i] - fineTmps[0]; // v_i - P R v_i
      deviation   = std::sqrt(norm2(fineTmps[1]) / norm2(_Aggregates.Subspace()[i]));

      LOG_MG_LVL << "Vector " << i << ": norm2(v_i) = " << norm2(_Aggregates.Subspace()[i])
                << " | norm2(R v_i) = " << norm2(coarseTmps[0]) << " | norm2(P R v_i) = " << norm2(fineTmps[0])
                << " | relative deviation = " << deviation;

      if(deviation > tolerance) {
        std::cout << " > " << tolerance << " -> check failed" << std::endl;
        abort();
      } else {
        std::cout << " < " << tolerance << " -> check passed" << std::endl;
      }
    }

    LOG_MG_LVL << "**************************************************" << std::endl;
    LOG_MG_LVL << "MG correctness check: 0 == (1 - R P) v_c" << std::endl;
    LOG_MG_LVL << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextLevel], coarseTmps[0]);

    _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]); //   P v_c
    _Aggregates.ProjectToSubspace(coarseTmps[1], fineTmps[0]);   // R P v_c

    coarseTmps[2] = coarseTmps[0] - coarseTmps[1]; // v_c - R P v_c
    deviation     = std::sqrt(norm2(coarseTmps[2]) / norm2(coarseTmps[0]));

    LOG_MG_LVL << "norm2(v_c) = " << norm2(coarseTmps[0])
              << " | norm2(R P v_c) = " << norm2(coarseTmps[1]) << " | norm2(P v_c) = " << norm2(fineTmps[0])
              << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    LOG_MG_LVL << "**************************************************" << std::endl;
    LOG_MG_LVL << "MG correctness check: 0 == (R D P - D_c) v_c" << std::endl;
    LOG_MG_LVL << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextLevel], coarseTmps[0]);

    _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]); //     P v_c
    _FineMdagMOp.Op(fineTmps[0], fineTmps[1]);                   //   D P v_c
    _Aggregates.ProjectToSubspace(coarseTmps[1], fineTmps[1]);   // R D P v_c

    _NextPreconditionerLevel->_FineMdagMOp.Op(coarseTmps[0], coarseTmps[2]); // D_c v_c

    coarseTmps[3] = coarseTmps[1] - coarseTmps[2]; // R D P v_c - D_c v_c
    deviation     = std::sqrt(norm2(coarseTmps[3]) / norm2(coarseTmps[1]));

    LOG_MG_LVL << "norm2(R D P v_c) = " << norm2(coarseTmps[1])
              << " | norm2(D_c v_c) = " << norm2(coarseTmps[2]) << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    LOG_MG_LVL << "**************************************************" << std::endl;
    LOG_MG_LVL << "MG correctness check: 0 == |(Im(v_c^dag D_c^dag D_c v_c)|" << std::endl;
    LOG_MG_LVL << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextLevel], coarseTmps[0]);

    _NextPreconditionerLevel->_FineMdagMOp.Op(coarseTmps[0], coarseTmps[1]);    //         D_c v_c
    _NextPreconditionerLevel->_FineMdagMOp.AdjOp(coarseTmps[1], coarseTmps[2]); // D_c^dag D_c v_c

    auto dot  = innerProduct(coarseTmps[0], coarseTmps[2]); //v_c^dag D_c^dag D_c v_c
    deviation = std::abs(imag(dot)) / std::abs(real(dot));

    LOG_MG_LVL << "Re(v_c^dag D_c^dag D_c v_c) = " << real(dot)
              << " | Im(v_c^dag D_c^dag D_c v_c) = " << imag(dot) << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    _NextPreconditionerLevel->runChecks(tolerance);
  }

  void reportTimings() {
    if(_Level == 0) printProfiling(LVL_STR + "AllLevels.Op.Global", "solve");
  }

  /////////////////////////////////////////////
  // Member Functions (Helper)
  /////////////////////////////////////////////

public:

  void recreateOperators(bool recurseDown) {
    _Prof.Start("ThisLevel.RecreateOperators.Intergrid");
#if defined(USE_NEW_COARSENING)
    _Aggregates.Create(_BasisVectors);
#else
    _Aggregates.Create(_BasisVectors());
#endif
    _Prof.Stop("ThisLevel.RecreateOperators.Intergrid");

    _Prof.Start("ThisLevel.RecreateOperators.CoarseOp");
    _CoarseMatrix.CoarsenOperator(
                                  _LevelInfo.Grids[_Level],
                                  _FineMdagMOp,
                                  _Aggregates);
    _Prof.Stop("ThisLevel.RecreateOperators.CoarseOp");

    LOG_MG_LVL << "Recreated intergrid operators and coarse operator" << std::endl;

    if(recurseDown) _NextPreconditionerLevel->recreateOperators(recurseDown);
  }

  void updateBasisVector(bool recurseDown, int activationLevel, int n) {
    if(activationLevel != _Level) { // not needed on level that triggered it since this has already been done
      _Prof.Start("ThisLevel.UpdateBasisVec");
      _BasisVectors()[n]    = _FineSol;

      // auto       normBefore = norm2(_BasisVectors()[n]);
      // FineVector tmp(_BasisVectors()[n]);
      // auto       scale   = std::pow(norm2(_FineSol), -0.5);
      // _BasisVectors()[n] = scale * _FineSol;
      // auto normAfter     = norm2(_BasisVectors()[n]);
      // tmp                = tmp - _BasisVectors()[n];
      // auto normDiff      = norm2(tmp);

      // LOG_MG_LVL << "Updated vector " << n << ":"
      //            << " normBefore = " << normBefore
      //            << " normAfter = " << normAfter
      //            << " normDiff.abs = " << normDiff
      //            << " normDiff.rel = " << normDiff / normBefore << std::endl;
      _Prof.Stop("ThisLevel.UpdateBasisVec");
    }

    if(recurseDown) _NextPreconditionerLevel->updateBasisVector(recurseDown, activationLevel, n);
  }

  std::map<std::string, MGProfileResult> getProfile() {
    auto lowerLevelProfiles     = _NextPreconditionerLevel->getProfile();
    auto myProfile              = _Prof.GetResults(LVL_STR);
    auto basisVecsProfile       = _BasisVectors.GetProfile(LVL_STR + "BVecs.");
    auto aggregatesProfile      = _Aggregates.GetProfile(LVL_STR + "Aggs.");
    auto coarsenedMatrixProfile = _CoarseMatrix.GetProfile(LVL_STR + "CMat.");

    _Prof.ResetAll();
    _BasisVectors.ResetProfile();
    _Aggregates.ResetProfile();
    _CoarseMatrix.ResetProfile();

    myProfile.insert(basisVecsProfile.begin(), basisVecsProfile.end());
    myProfile.insert(aggregatesProfile.begin(), aggregatesProfile.end());
    myProfile.insert(coarsenedMatrixProfile.begin(), coarsenedMatrixProfile.end());
    myProfile.insert(lowerLevelProfiles.begin(), lowerLevelProfiles.end());
    return myProfile;
  }

private:

  template<int ncl = nCoarserLevels, typename std::enable_if<(ncl >= 2), int>::type = 0>
  void projectBasisVectorsDownwardsIfNecessary() {
    _Prof.Start("ThisLevel.ProjectBasisVecsDown");
    LOG_MG_LVL << "Projecting basis vectors to next coarser level" << std::endl;
    for(int n = 0; n < nB; n++) {
      _Aggregates.ProjectToSubspace(_NextPreconditionerLevel->_BasisVectors()[n], _BasisVectors()[n]);
    }
    _Prof.Stop("ThisLevel.ProjectBasisVecsDown");
  }

  template<int ncl = nCoarserLevels, typename std::enable_if<(ncl <= 1), int>::type = 0>
  void projectBasisVectorsDownwardsIfNecessary() {
    LOG_MG_LVL << "NOT projecting basis vectors to next coarser level" << std::endl;
  }

  RealD calcResidual(LinearOperatorBase<FineVector>& linOp, FineVector const& in, FineVector const& vec) {
    _Prof.Start("ThisLevel.Op.CalcResidual");
    FineVector tmp(in.Grid());
    linOp.Op(vec, tmp);
    tmp = in - tmp;
    auto nrm = norm2(tmp);
    _Prof.Stop("ThisLevel.Op.CalcResidual");
    return nrm;
  }

  void printInfo(RealD inputNorm, RealD cgcResidual, RealD postSmootherResidual) {
    LOG_MG_LVL << _Params.cyclingStrategy[_Level]
               << ": Input norm = " << sqrt(inputNorm)
               << " rel Coarse residual = " << sqrt(cgcResidual / inputNorm)
               << " rel Post-Smoother residual = "
               << sqrt(postSmootherResidual / inputNorm) << std::endl;
  }

  void printProfiling(std::string const& totalStr, std::string const& prefix = "") {
    std::cout << GridLogPerformance << "= Profiling =============================================" << std::endl;
    auto prof = getProfile();
    prettyPrintProfiling(prefix, prof, prof[totalStr].t);
    std::cout << GridLogPerformance << "=========================================================" << std::endl;
  }
};

// Specialization for the coarsest level
template<class Fobj, class CComplex, int nBasis, int nCoarsespins, class Matrix>
class MultiGridPreconditioner<Fobj, CComplex, nBasis, nCoarsespins, 0, Matrix> : public MultiGridPreconditionerBase<Lattice<Fobj>> {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  typedef Matrix        FineDiracMatrix;
  typedef Lattice<Fobj> FineVector;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

  int _Level;

  MultiGridParams& _Params;
  LevelInfo&       _LevelInfo;

  FineDiracMatrix& _FineMatrix;
  FineDiracMatrix& _SmootherMatrix;

  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineMdagMOp;
  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineSmootherMdagMOp;

  FineVector _FineSrc;
  FineVector _FineSol;

  Profiler _Prof;

  /////////////////////////////////////////////
  // Member Functions (Interface)
  /////////////////////////////////////////////

  MultiGridPreconditioner(
    MultiGridParams& Params,
    LevelInfo&       LvlInfo,
    FineDiracMatrix& FineMat,
    FineDiracMatrix& SmootherMat)
    : _Level(Params.nLevels - (0 + 1))
    , _Params(Params)
    , _LevelInfo(LvlInfo)
    , _FineMatrix(FineMat)
    , _SmootherMatrix(SmootherMat)
    , _FineMdagMOp(_FineMatrix)
    , _FineSmootherMdagMOp(_SmootherMatrix)
    , _FineSrc(_LevelInfo.Grids[_Level])
    , _FineSol(_LevelInfo.Grids[_Level]) {

    LOG_MG_LVL << "Will not be doing any subspace generation since this is the coarsest level" << std::endl;
  }

  void initialSetup() {}
  void refinementSetup() {}

  virtual void operator()(FineVector const& in, FineVector& out) {
    _Prof.Start("ThisLevel.Op.Total");
    _Prof.Start("ThisLevel.Op.Misc");
    conformable(_LevelInfo.Grids[_Level], in.Grid());
    conformable(in, out);

    TrivialPrecon<FineVector> simplePrec;
    auto                      coarsestSolver =
      createUtilisedSolver(_Params.coarsestSolverType, _Params.coarsestSolverParams, simplePrec);
    auto coarsestRBSolver = createUtilisedRBSolver(_Params.coarsestSolverType, *coarsestSolver);
    _Prof.Stop("ThisLevel.Op.Misc");

    _Prof.Start("ThisLevel.Op.Smoother");
    if(_Params.coarsestSolverParams.useRBPrec)
      (*coarsestRBSolver)(_FineMatrix, in, out);
    else
      (*coarsestSolver)(_FineMdagMOp, in, out);
    _Prof.Stop("ThisLevel.Op.Smoother");
    _Prof.Stop("ThisLevel.Op.Total");
  }

  void runChecks(RealD tolerance) {}
  void reportTimings() {}

  /////////////////////////////////////////////
  // Member Functions (Helper)
  /////////////////////////////////////////////

public:

  void recreateOperators(bool recurseDown) {}
  void updateBasisVector(bool recurseDown, int activationLevel, int n) {}

  std::map<std::string, MGProfileResult> getProfile() {
    auto myProfile = _Prof.GetResults(LVL_STR);
    _Prof.ResetAll();
    return myProfile;
  }
};

template<class Fobj, class CComplex, int nBasis, int nCoarseSpins, int nLevels, class Matrix>
using NLevelMGPreconditioner = MultiGridPreconditioner<Fobj, CComplex, nBasis, nCoarseSpins, nLevels - 1, Matrix>;

template<class Fobj, class CComplex, int nBasis, int nCoarseSpins, class Matrix>
std::unique_ptr<MultiGridPreconditionerBase<Lattice<Fobj>>>
createMGInstance(MultiGridParams& Params, LevelInfo& levelInfo, Matrix& FineMat, Matrix& SmootherMat) {

#define CASE_FOR_N_LEVELS(nLevels) \
  case nLevels: \
    return std::unique_ptr<NLevelMGPreconditioner<Fobj, CComplex, nBasis, nCoarseSpins, nLevels, Matrix>>( \
      new NLevelMGPreconditioner<Fobj, CComplex, nBasis, nCoarseSpins, nLevels, Matrix>( \
        Params, levelInfo, FineMat, SmootherMat)); \
    break;

  switch(Params.nLevels) {
    CASE_FOR_N_LEVELS(2);
    CASE_FOR_N_LEVELS(3);
    CASE_FOR_N_LEVELS(4);
    default:
      std::cout << GridLogError << "We currently only support nLevels ∈ {2, 3, 4}" << std::endl;
      exit(EXIT_FAILURE);
      break;
  }
#undef CASE_FOR_N_LEVELS
}

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
