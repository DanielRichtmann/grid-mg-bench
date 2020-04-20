/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/AggregationRework.h

    Copyright (C) 2015 - 2019

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

template<class CoarseningPolicy>
class Aggregation : public CoarseningPolicy, public Profileable {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
  INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);

  using Kernels               = AggregationKernels<CoarseningPolicy>;
  using CoarseningLookupTable = Grid::Rework::CoarseningLookupTable;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:
  GridBase*             coarseGrid_;
  GridBase*             fineGrid_;
  IntergridOperator     subspace_;
  int                   checkerBoard_;
  int                   useFastProjects_;
  CoarseningLookupTable lut_;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:
  Aggregation(GridBase* CoarseGrid, GridBase* FineGrid, int CheckerBoard, int UseFastProjects = 0)
    : coarseGrid_(CoarseGrid)
    , fineGrid_(FineGrid)
    , subspace_(Nc_c, FineGrid)
    , checkerBoard_(CheckerBoard)
    , useFastProjects_(UseFastProjects)
    , lut_(coarseGrid_, fineGrid_) {
    subdivides(coarseGrid_, fineGrid_);
    std::cout << GridLogDebug << "Constructed Rework::Aggregation with fast projects set to " << useFastProjects_ << std::endl;
  }

  IntergridOperator&       Subspace() { return subspace_; }
  IntergridOperator const& Subspace() const { return subspace_; }

  void Orthogonalise(int checkOrthogonality = 1, int passes = 2) {
    prof_.Start("Create.Orthogonalise");
    ScalarField InnerProd(coarseGrid_);
    for(int n = 0; n < passes; ++n) {
      std::cout << GridLogMessage << "Gram-Schmidt pass " << n + 1 << std::endl;
      Kernels::aggregateOrthogonalise(InnerProd, subspace_, lut_);
    }
    prof_.Stop("Create.Orthogonalise");
    if(checkOrthogonality) CheckOrthogonal();
  }

  void CheckOrthogonal() {
    prof_.Start("Create.CheckOrthogonal");
    std::cout << GridLogMessage << "Gram-Schmidt checking orthogonality" << std::endl;
    FermionField iProj(coarseGrid_);
    FermionField eProj(coarseGrid_);
    for(int i = 0; i < Nc_c; ++i) {
      ProjectToSubspace(iProj, subspace_[i]);
      eProj        = Zero();
      auto eProj_v = eProj.View();
      accelerator_for(ss, coarseGrid_->oSites(), Simd::Nsimd(), {
        for(int s = 0; s < Ns_c; ++s) eProj_v[ss]()(s)(i) = Simd(1.0);
      });
      eProj = eProj - iProj;
      std::cout << GridLogMessage << "Orthog check error " << i << " " << norm2(eProj) << std::endl;
    }
    std::cout << GridLogMessage << "Orthog check done" << std::endl;
    prof_.Stop("Create.CheckOrthogonal");
  }

  void ProjectToSubspace(FermionField& CoarseVec, FineFermionField const& FineVec) const {
    if(useFastProjects_)
      Kernels::aggregateProjectFast(CoarseVec, FineVec, subspace_, lut_); // use internal lut
    else
      Kernels::aggregateProject(CoarseVec, FineVec, subspace_, lut_); // use internal lut
  }

  void ProjectToSubspace(FermionField&                CoarseVec,
                         FineFermionField const&      FineVec,
                         CoarseningLookupTable const& Lut) const {
    if(useFastProjects_)
      Kernels::aggregateProjectFast(CoarseVec, FineVec, subspace_, Lut); // use external lut
    else
      Kernels::aggregateProject(CoarseVec, FineVec, subspace_, Lut); // use external lut
  }

  void PromoteFromSubspace(FermionField const& CoarseVec, FineFermionField& FineVec) const {
    if(useFastProjects_)
      Kernels::aggregatePromoteFast(CoarseVec, FineVec, subspace_, lut_); // use internal lut
    else
      Kernels::aggregatePromote(CoarseVec, FineVec, subspace_, lut_); // use internal lut
  }

  void Create(MGBasisVectors<FineFermionField> const& BasisVectors) {
    // this function generates the MG intergrid operator from the MG basis vectors
    // these are separate since we need the basis vectors externally for the refinement setup
    prof_.Start("Create.Total");
    prof_.Start("Create.Copy");
    assert(BasisVectors().size() == Nc_c);
    for(int n = 0; n < Nc_c; ++n)
      subspace_[n] = BasisVectors()[n];
    prof_.Stop("Create.Copy");

    // DoChiralDoubling(); // TODO: do we want that here? Could be necessary for dwf, but I think this can also be done during coarse operator construction
    Orthogonalise();
    prof_.Stop("Create.Total");
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
