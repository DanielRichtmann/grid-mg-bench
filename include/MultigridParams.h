/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/MultiGridParams.h

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
GRID_SERIALIZABLE_ENUM(MGBasisVectorsType, undef,
                       nullVectors, 1,
                       testVectors, 2
);
GRID_SERIALIZABLE_ENUM(MGCyclingStrategy, undef,
                       VCYCLE, 1,
                       WCYCLE, 2,
                       KCYCLE, 3
);
GRID_SERIALIZABLE_ENUM(MGUtilizedSolverType, undef,
                       MR, 1,
                       GMRES, 2,
                       FGMRES, 3,
                       BiCGSTAB, 4
);

struct MultiGridUtilizedSolverParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MultiGridUtilizedSolverParams,
    RealD, tolerance,     // relative tolerance goal
    int,   maxIter,       // maximum number of iterations
    int,   restartLength, // for restarted solvers, ignored for simple ones
    bool,  useRBPrec      // use the rb preconditioned version of the solver
  );

  MultiGridUtilizedSolverParams(
    RealD tolerance_     = 1.0e-3,
    int   maxIter_       = 1,
    int   restartLength_ = 10,
    bool  useRBPrec_     = false
  )
    : tolerance(tolerance_)
    , maxIter(maxIter_)
    , restartLength(restartLength_)
    , useRBPrec(useRBPrec_)
  {}
};

struct MultiGridParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(MultiGridParams,
    int,                                        nLevels,                      // number of levels in the MG method
    std::vector<std::vector<int>>,              blockSizes,                   // block sizes of the lattice blocks that are combined into one coarse site
    std::vector<int>,                           setupInitialIter,             // number of initial setup iterations (Osborn)
    std::vector<int>,                           setupRefinementIter,          // number of iterative refinement setup iterations (Wuppertal/Lüscher)
    MGBasisVectorsType,                         setupBasisVectorsType,        // type of MG basis vectors to use (test vectors or null vectors)
    bool,                                       setupPreOrthonormalise,       // pre orthonormalise MG basis vectors in setup phase
    bool,                                       setupPostOrthonormalise,      // post orthonormalise MG basis vectors in setup phase
    bool,                                       setupGenerateOnAllLevels,     // generate MG basis vectors on all levels or generate only on finest and project downwards (initial setup)
    bool,                                       setupRecursiveUpdates,        // update MG basis vectors on coarser levels with results obtained while constructing a finer level (refinement setup)
    std::vector<MGCyclingStrategy>,             cyclingStrategy,              // cycling strategy to use
    std::vector<MGUtilizedSolverType>,          setupSolverType,              // solver type to use in the setup on each level (i.e., the initial setup phase)
    std::vector<MultiGridUtilizedSolverParams>, setupSolverParams,            // params to the setup solver (i.e., in the initial setup phase)
    std::vector<MGUtilizedSolverType>,          wrapperSolverType,            // solver type to use as the wrapper (= k-cycle) solver
    std::vector<MultiGridUtilizedSolverParams>, wrapperSolverParams,          // parameters to the wrapper solver
    std::vector<MGUtilizedSolverType>,          smootherSolverType,           // solver type to use as the smoother on each level
    std::vector<MultiGridUtilizedSolverParams>, smootherSolverParams,         // parameters to the smoother solver
    MGUtilizedSolverType,                       coarsestSolverType,           // solver type to use on the coarsest level
    MultiGridUtilizedSolverParams,              coarsestSolverParams          // parameters to the coarsest level solver
  );

  MultiGridParams(
    int                                        nLevels_                      = 3,
    std::vector<std::vector<int>>              blockSizes_                   = {{4,4,4,4}, {2,2,2,2}},
    std::vector<int>                           setupInitialIter_             = {1, 1},
    std::vector<int>                           setupRefinementIter_          = {0, 0},
    MGBasisVectorsType                         setupBasisVectorsType_        = MGBasisVectorsType::testVectors,
    bool                                       setupPreOrthonormalise_       = true,
    bool                                       setupPostOrthonormalise_      = true,
    bool                                       setupGenerateOnAllLevels_     = true,
    bool                                       setupRecursiveUpdates_        = true,
    std::vector<MGCyclingStrategy>             cyclingStrategy_              = {MGCyclingStrategy::KCYCLE, MGCyclingStrategy::KCYCLE},
    std::vector<MGUtilizedSolverType>          setupSolverType_              = {MGUtilizedSolverType::BiCGSTAB, MGUtilizedSolverType::BiCGSTAB},
    std::vector<MultiGridUtilizedSolverParams> setupSolverParams_            = {MultiGridUtilizedSolverParams(1.0e-3, 100, 20, false), MultiGridUtilizedSolverParams(1.0e-3, 100, 20, false)},
    std::vector<MGUtilizedSolverType>          wrapperSolverType_            = {MGUtilizedSolverType::FGMRES, MGUtilizedSolverType::FGMRES},
    std::vector<MultiGridUtilizedSolverParams> wrapperSolverParams_          = {MultiGridUtilizedSolverParams(1.0e-1, 10, 5, false), MultiGridUtilizedSolverParams(1.0e-1, 10, 5, false)},
    std::vector<MGUtilizedSolverType>          smootherSolverType_           = {MGUtilizedSolverType::GMRES, MGUtilizedSolverType::GMRES},
    std::vector<MultiGridUtilizedSolverParams> smootherSolverParams_         = {MultiGridUtilizedSolverParams(1.0e-3, 4, 4, false), MultiGridUtilizedSolverParams(1.0e-3, 4, 4, false)},
    MGUtilizedSolverType                       coarsestSolverType_           = MGUtilizedSolverType::GMRES,
    MultiGridUtilizedSolverParams              coarsestSolverParams_         = MultiGridUtilizedSolverParams(5.0e-2, 50, 10, false)
  )
    : nLevels(nLevels_)
    , blockSizes(blockSizes_)
    , setupInitialIter(setupInitialIter_)
    , setupRefinementIter(setupRefinementIter_)
    , setupBasisVectorsType(setupBasisVectorsType_)
    , setupPreOrthonormalise(setupPreOrthonormalise_)
    , setupPostOrthonormalise(setupPostOrthonormalise_)
    , setupGenerateOnAllLevels(setupGenerateOnAllLevels_)
    , setupRecursiveUpdates(setupRecursiveUpdates_)
    , cyclingStrategy(cyclingStrategy_)
    , setupSolverType(setupSolverType_)
    , setupSolverParams(setupSolverParams_)
    , wrapperSolverType(wrapperSolverType_)
    , wrapperSolverParams(wrapperSolverParams_)
    , smootherSolverType(smootherSolverType_)
    , smootherSolverParams(smootherSolverParams_)
    , coarsestSolverType(coarsestSolverType_)
    , coarsestSolverParams(coarsestSolverParams_)
  {}

  template <class ReaderClass>
  MultiGridParams(Reader<ReaderClass>& reader){
    read(reader, "Multigrid", *this);
  }
};

class MGTestOtherParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MGTestOtherParams,
    RealD,            outerSolverTol,
    Integer,          outerSolverMaxIter,
    Integer,          outerSolverRestartLength,
    RealD,            massSetup,
    RealD,            massSolve,
    RealD,            csw,
    std::string,      config,
    std::string,      sourceType,
    bool,             useAntiPeriodicBC
  );

  // constructor with default values
  MGTestOtherParams(
    RealD       outerSolverTol_           = 1e-12,
    Integer     outerSolverMaxIter_       = 100,
    Integer     outerSolverRestartLength_ = 20,
    RealD       massSetup_                = 0.1,
    RealD       massSolve_                = 0.1,
    RealD       csw_                      = 1.0,
    std::string config_                   = "foo",
    std::string sourceType_               = "random",
    bool        useAntiPeriodicBC_        = false
  )
    : outerSolverTol(outerSolverTol_)
    , outerSolverMaxIter(outerSolverMaxIter_)
    , outerSolverRestartLength(outerSolverRestartLength_)
    , massSetup(massSetup_)
    , massSolve(massSolve_)
    , csw(csw_)
    , config(config_)
    , sourceType(sourceType_)
    , useAntiPeriodicBC(useAntiPeriodicBC_)
  {}
};

class MGTestParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MGTestParams,
    MultiGridParams,   mg,
    MGTestOtherParams, test
  );
};
// clang-format on

void checkParameterValidity(MultiGridParams const& params) {

  auto correctSize = params.nLevels - 1;

  assert(correctSize == params.blockSizes.size());
  assert(correctSize == params.setupInitialIter.size());
  assert(correctSize == params.setupRefinementIter.size());
  assert(correctSize == params.cyclingStrategy.size());
  assert(correctSize == params.setupSolverType.size());
  assert(correctSize == params.setupSolverParams.size());
  assert(correctSize == params.wrapperSolverType.size());
  assert(correctSize == params.wrapperSolverParams.size());
  assert(correctSize == params.smootherSolverType.size());
  assert(correctSize == params.smootherSolverParams.size());

  for(int l = 0; l < correctSize; ++l) {
    assert(params.cyclingStrategy[l] != MGCyclingStrategy::undef);
    assert(params.cyclingStrategy[l] != MGCyclingStrategy::WCYCLE); // not supported at the moment -> TODO
    assert(params.setupSolverType[l] != MGUtilizedSolverType::undef);
    if(params.cyclingStrategy[l] == MGCyclingStrategy::KCYCLE)
      assert(params.wrapperSolverType[l] != MGUtilizedSolverType::undef);
    assert(params.smootherSolverType[l] != MGUtilizedSolverType::undef);
    assert(params.wrapperSolverParams[l].useRBPrec != true);  // not supported at the moment -> TODO
  }

  assert(params.setupBasisVectorsType != MGBasisVectorsType::undef);
  assert(params.coarsestSolverType != MGUtilizedSolverType::undef);
}

void checkParameterValidity(MGTestOtherParams const& params) {
  assert(params.sourceType == "ones" || params.sourceType == "random" || params.sourceType == "gaussian");
}

void checkParameterValidity(MGTestParams const& params) {
  checkParameterValidity(params.mg);
  checkParameterValidity(params.test);
}

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
