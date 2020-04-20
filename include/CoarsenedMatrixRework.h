/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/CoarsenedMatrixRework.h

    Copyright (C) 2015 - 2019

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>
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
class CoarsenedMatrix : public CheckerBoardedSparseMatrixBase<typename CoarseningPolicy::FermionField>
                      , public CoarseningPolicy
                      , public Profileable {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
  INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);

  using Kernels               = CoarsenedMatrixKernels<CoarseningPolicy>;
  using CoarseningLookupTable = Grid::Rework::CoarseningLookupTable;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

  GridBase* grid_;
  GridBase* cbgrid_;

  GeometryDevelop geom_;

  int speedLevel_;
  int hermitian_;

  CartesianStencil<SiteSpinor, SiteSpinor, int> stencil_;
  CartesianStencil<SiteSpinor, SiteSpinor, int> stencilEven_;
  CartesianStencil<SiteSpinor, SiteSpinor, int> stencilOdd_;

  std::vector<LinkField> Y_; // Y = Kate's notation
  std::vector<LinkField> YEven_;
  std::vector<LinkField> YOdd_;

  LinkField selfStencilLinkInv_;
  LinkField selfStencilLinkInvEven_;
  LinkField selfStencilLinkInvOdd_;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

  CoarsenedMatrix(GridCartesian&         CoarseGrid,
                  GridRedBlackCartesian& CoarseRBGrid,
                  int                    speedLevel = 0,
                  int                    hermitian  = 0)
    : grid_(&CoarseGrid)
    , cbgrid_(&CoarseRBGrid)
    , geom_(CoarseGrid._ndimension, true)
    , speedLevel_(speedLevel)
    , hermitian_(hermitian)
    , stencil_(&CoarseGrid, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    , stencilEven_(&CoarseRBGrid, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    , stencilOdd_(&CoarseRBGrid, geom_.npoint, Odd, geom_.directions, geom_.displacements, 0)
    , Y_(geom_.npoint, &CoarseGrid)
    , YEven_(geom_.npoint, &CoarseRBGrid)
    , YOdd_(geom_.npoint, &CoarseRBGrid)
    , selfStencilLinkInv_(&CoarseGrid)
    , selfStencilLinkInvEven_(&CoarseRBGrid)
    , selfStencilLinkInvOdd_(&CoarseRBGrid) {
    std::cout << GridLogDebug << "Constructed Rework::CoarsenedMatrix with speed level " << speedLevel_ << std::endl;
  }

  GridBase* Grid() { return grid_; };

  GridBase* RedBlackGrid() { return cbgrid_; };

  int ConstEE() { return 0; }

  RealD M(FermionField const& in, FermionField& out) {
    conformable(Grid(), in.Grid());
    conformable(in.Grid(), out.Grid());

    SimpleCompressor<SiteSpinor> compressor;
    stencil_.HaloExchange(in, compressor);

    auto in_v  = in.View();
    auto out_v = out.View();

    typedef LatticeView<SiteLinkField>       LinkFieldView;
    typedef decltype(coalescedRead(in_v[0])) SiteSpinorCR;

    // need to access data per raw pointer on GPU
    auto  Y_v_c   = getViewContainer(Y_);
    auto* Y_v_c_p = &Y_v_c[0];

    // need to take references, otherwise we get illegal memory accesses
    // happens since the lambda copies the this pointer which points to host memory, see
    // - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#star-this-capture
    // - https://devblogs.nvidia.com/new-compiler-features-cuda-8/
    GeometryDevelop& geom = geom_;
    CartesianStencil<SiteSpinor, SiteSpinor, int>& stencil = stencil_;

    accelerator_for(ss, Grid()->oSites(), Simd::Nsimd(), {
      SiteSpinorCR  res = Zero();
      SiteSpinorCR  nbr;
      int           lane = SIMTlane(Simd::Nsimd());
      int           ptype;
      StencilEntry* SE;

      for(int point = 0; point < geom.npoint; ++point) {
        SE = stencil.GetEntry(ptype, point, ss);

        if(SE->_is_local)
          nbr = coalescedReadPermute(in_v[SE->_offset], ptype, SE->_permute, lane);
        else
          nbr = coalescedRead(stencil.CommBuf()[SE->_offset], lane);

        synchronise();

        res = res + coalescedRead(Y_v_c_p[point][ss]) * nbr;
      }
      coalescedWrite(out_v[ss], res, lane);
    });
    return norm2(out);
  }

  RealD Mdag(FermionField const& in, FermionField& out) {
    if(hermitian_) {
      // corresponds to Petrov-Galerkin coarsening
      return M(in, out);
    } else {
      // corresponds to Galerkin coarsening
      FermionField tmp(Grid());
      G5C(tmp, in); // NOTE: This is actually only valid for the two spin version
      M(tmp, out);
      G5C(out, out);
      return norm2(out);
    }
  }

  void Mdiag(FermionField const& in, FermionField& out) {
    Mooee(in, out); // just like for fermion operators
  }

  void Mdir(FermionField const& in, FermionField& out, int dir, int disp) {
    DhopDir(in, out, dir, disp);
  }

  void Meooe(FermionField const& in, FermionField& out) {
    if(in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerNo);
    } else {
      DhopOE(in, out, DaggerNo);
    }
  }

  void MeooeDag(FermionField const& in, FermionField& out) {
    if(in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerYes);
    } else {
      DhopOE(in, out, DaggerYes);
    }
  }

  void Mooee(FermionField const& in, FermionField& out) { MooeeInternal(in, out, DaggerNo, InverseNo); }

  void MooeeInv(FermionField const& in, FermionField& out) { MooeeInternal(in, out, DaggerNo, InverseYes); }

  void MooeeDag(FermionField const& in, FermionField& out) { MooeeInternal(in, out, DaggerYes, InverseNo); }

  void MooeeInvDag(FermionField const& in, FermionField& out) { MooeeInternal(in, out, DaggerYes, InverseYes); }

  void Dhop(FermionField const& in, FermionField& out, int dag) {
    conformable(in.Grid(), Grid()); // verifies full grid
    conformable(in.Grid(), out.Grid());

    out.Checkerboard() = in.Checkerboard();

    DhopInternal(stencil_, Y_, in, out, dag);
  }

  void DhopOE(FermionField const& in, FermionField& out, int dag) {
    conformable(in.Grid(), RedBlackGrid()); // verifies half grid
    conformable(in.Grid(), out.Grid());     // drops the cb check

    assert(in.Checkerboard() == Even);
    out.Checkerboard() = Odd;

    DhopInternal(stencilEven_, YOdd_, in, out, dag);
  }

  void DhopEO(FermionField const& in, FermionField& out, int dag) {
    conformable(in.Grid(), RedBlackGrid()); // verifies half grid
    conformable(in.Grid(), out.Grid());     // drops the cb check

    assert(in.Checkerboard() == Odd);
    out.Checkerboard() = Even;

    DhopInternal(stencilOdd_, YEven_, in, out, dag);
  }

  void DhopDir(FermionField const& in, FermionField& out, int dir, int disp) {
    conformable(Grid(), in.Grid()); // verifies full grid
    conformable(in.Grid(), out.Grid());

    SimpleCompressor<SiteSpinor> compressor;
    stencil_.HaloExchange(in, compressor);

    auto point = geom_.PointFromDirDisp(dir, disp);

    auto in_v      = in.View();
    auto out_v     = out.View();
    auto Y_point_v = Y_[point].View();

    typedef decltype(coalescedRead(in_v[0])) SiteSpinorCR;

    // need to take a reference, otherwise we get illegal memory accesses
    // see links above for reason
    CartesianStencil<SiteSpinor, SiteSpinor, int>& stencil = stencil_;

    accelerator_for(ss, Grid()->oSites(), Simd::Nsimd(), {
      SiteSpinorCR    res = Zero();
      SiteSpinorCR    nbr;
      int             lane = SIMTlane(Simd::Nsimd());
      int             ptype;
      StencilEntry*   SE;

      SE = stencil.GetEntry(ptype, point, ss);

      if(SE->_is_local)
        nbr = coalescedReadPermute(in_v[SE->_offset], ptype, SE->_permute, lane);
      else
        nbr = coalescedRead(stencil.CommBuf()[SE->_offset], lane);

      synchronise();

      res = res + Y_point_v(ss) * nbr;

      coalescedWrite(out_v[ss], res, lane);
    });
  }

  void DhopInternal(CartesianStencil<SiteSpinor, SiteSpinor, int>& st,
                    std::vector<LinkField>&                        Y,
                    FermionField const&                            in,
                    FermionField&                                  out,
                    int                                            dag) {
    if(dag) {
      FermionField tmp(in.Grid());
      tmp.Checkerboard() = in.Checkerboard();
      FermionField tmp2(out.Grid());
      tmp2.Checkerboard() = out.Checkerboard();
      G5C(tmp, in); // FIXME: This explicitly ties us to Galerkin coarsening

      SimpleCompressor<SiteSpinor> compressor;
      st.HaloExchange(tmp, compressor);

      auto tmp_v  = tmp.View();
      auto tmp2_v = tmp2.View();

      typedef LatticeView<SiteLinkField>        LinkFieldView;
      typedef decltype(coalescedRead(tmp_v[0])) SiteSpinorCR;

      // need to access data per raw pointer on GPU
      auto  Y_v_c   = getViewContainer(Y);
      auto* Y_v_c_p = &Y_v_c[0];

      // need to take a reference, otherwise we get illegal memory accesses
      // see links above for reason
      GeometryDevelop& geom = geom_;

      accelerator_for(ss, tmp.Grid()->oSites(), Simd::Nsimd(), {
        SiteSpinorCR    res = Zero();
        SiteSpinorCR    nbr;
        int             lane = SIMTlane(Simd::Nsimd());
        int             ptype;
        StencilEntry*   SE;

        for(int point = 0; point < geom.npoint; ++point) {
          if(point != geom.SelfStencilPoint()) {
            SE = st.GetEntry(ptype, point, ss);

            if(SE->_is_local)
              nbr = coalescedReadPermute(tmp_v[SE->_offset], ptype, SE->_permute, lane);
            else
              nbr = coalescedRead(st.CommBuf()[SE->_offset], lane);

            synchronise();

            res = res + coalescedRead(Y_v_c_p[point][ss]) * nbr;
          }
        }
        coalescedWrite(tmp2_v[ss], res, lane);
      });
      G5C(out, tmp2); // FIXME: This explicitly ties us to Galerkin coarsening
    } else {
      SimpleCompressor<SiteSpinor> compressor;
      st.HaloExchange(in, compressor);

      auto in_v  = in.View();
      auto out_v = out.View();

      typedef LatticeView<SiteLinkField>       LinkFieldView;
      typedef decltype(coalescedRead(in_v[0])) SiteSpinorCR;

      // need to access data per raw pointer on GPU
      auto  Y_v_c   = getViewContainer(Y);
      auto* Y_v_c_p = &Y_v_c[0];

      // need to take a reference, otherwise we get illegal memory accesses
      // see links above for reason
      GeometryDevelop& geom = geom_;

      accelerator_for(ss, in.Grid()->oSites(), Simd::Nsimd(), {
        SiteSpinorCR    res = Zero();
        SiteSpinorCR    nbr;
        int             lane = SIMTlane(Simd::Nsimd());
        int             ptype;
        StencilEntry*   SE;

        for(int point = 0; point < geom.npoint; ++point) {
          if(point != geom.SelfStencilPoint()) {
            SE = st.GetEntry(ptype, point, ss);

            if(SE->_is_local)
              nbr = coalescedReadPermute(in_v[SE->_offset], ptype, SE->_permute, lane);
            else
              nbr = coalescedRead(st.CommBuf()[SE->_offset], lane);

            synchronise();

            res = res + coalescedRead(Y_v_c_p[point][ss]) * nbr;
          }
        }
        coalescedWrite(out_v[ss], res, lane);
      });
    }
  }

  void CoarsenOperator(GridBase*                             FineGrid,
                       LinearOperatorBase<FineFermionField>& LinOp,
                       Aggregation<CoarseningPolicy>&        Projector) {
    prof_.Start("CoarsenOperator.Total");
    prof_.Start("CoarsenOperator.Misc");
    std::vector<FineFermionField> phiSplit(Ns_c, FineGrid);
    std::vector<FineFermionField> MphiSplit(geom_.npoint * Ns_c, FineGrid);
    std::vector<FineFermionField> MphiSplit_e(Ns_c, FineGrid);
    std::vector<FineFermionField> MphiSplit_o(Ns_c, FineGrid);
    std::vector<FineFermionField> iSum(Ns_c, FineGrid);
    std::vector<FineFermionField> calcTmp(Ns_c, FineGrid);

    std::vector<FermionField> iProjSplit(Ns_c, Grid());
    std::vector<FermionField> oProjSplit(Ns_c, Grid());

    LatticeInteger blockCB(FineGrid);
    LatticeInteger coor(FineGrid);

    // clang-format off
    FineScalarField one(FineGrid);  one  = 1;
    FineScalarField zero(FineGrid); zero = Zero();
    FineScalarField outerSites(FineGrid);
    FineScalarField evenBlocks(FineGrid);
    FineScalarField oddBlocks(FineGrid);
    // clang-format on

    std::vector<FineScalarField> innerSites(geom_.npoint, FineGrid);

    std::vector<CoarseningLookupTable> iLut(geom_.npoint);
    std::vector<CoarseningLookupTable> oLut(geom_.npoint);

    auto self_stencil = geom_.SelfStencilPoint();
    for(int p = 0; p < geom_.npoint; ++p) Y_[p] = Zero();
    prof_.Stop("CoarsenOperator.Misc");

    prof_.Start("CoarsenOperator.SetupLuts");
    for(int p = 0; p < geom_.npoint; ++p) {
      int dir  = geom_.directions[p];
      int disp = geom_.displacements[p];

      LatticeCoordinate(coor, dir);

      Integer block = (FineGrid->_rdimensions[dir]) / (Grid()->_rdimensions[dir]);
      iLut[p].populate(Grid(), FineGrid);
      oLut[p].populate(Grid(), FineGrid);

      if(disp == 0)       innerSites[p] = one;
      else if(disp == +1) innerSites[p] = where(mod(coor, block) != (block - 1), one, zero);
      else if(disp == -1) innerSites[p] = where(mod(coor, block) != (Integer)0,  one, zero);
      else assert(0);

      outerSites = one - innerSites[p];

      iLut[p].deleteUnneededFineSites(innerSites[p]);
      oLut[p].deleteUnneededFineSites(outerSites);

      if(disp == -1) blockCB = blockCB + div(coor, block);
    }
    evenBlocks = where(mod(blockCB, 2) == (Integer)0, one, zero);
    oddBlocks  = one - evenBlocks;
    prof_.Stop("CoarsenOperator.SetupLuts");

#define k_loop_Ns_c for(int k = 0; k < Ns_c; ++k)

    for(int i = 0; i < Nc_c; ++i) {
      prof_.Start("CoarsenOperator.ExtractSpinComponents");
      Kernels::extractSpinComponents(Projector.Subspace()[i], phiSplit);
      prof_.Stop("CoarsenOperator.ExtractSpinComponents");

      std::cout << GridLogMessage << "(" << i << ") .." << std::endl;

      if(speedLevel_ == 0) { // no optimizations
        prof_.Start("CoarsenOperator.ApplyOp");
        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          if(disp == 0) k_loop_Ns_c LinOp.OpDiag(phiSplit[k], MphiSplit[p * Ns_c + k]);
          else          k_loop_Ns_c LinOp.OpDir(phiSplit[k], MphiSplit[p * Ns_c + k], dir, disp);
        }
        prof_.Stop("CoarsenOperator.ApplyOp", Ns_c);

        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          prof_.Start("CoarsenOperator.ProjectToSubspace");
          k_loop_Ns_c Projector.ProjectToSubspace(iProjSplit[k], MphiSplit[p * Ns_c + k], iLut[p]);
          k_loop_Ns_c Projector.ProjectToSubspace(oProjSplit[k], MphiSplit[p * Ns_c + k], oLut[p]);
          prof_.Stop("CoarsenOperator.ProjectToSubspace", Ns_c);

          prof_.Start("CoarsenOperator.ConstructLinksFull");
          Kernels::constructLinksFull(i, p, disp, self_stencil, Y_, iProjSplit, oProjSplit);
          prof_.Stop("CoarsenOperator.ConstructLinksFull");
        }
      } else if(speedLevel_ == 1) { // save projects by summing up inner contributions before projecting
        prof_.Start("CoarsenOperator.ApplyOp");
        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          if(disp == 0) k_loop_Ns_c LinOp.OpDiag(phiSplit[k], MphiSplit[p * Ns_c + k]);
          else          k_loop_Ns_c LinOp.OpDir(phiSplit[k], MphiSplit[p * Ns_c + k], dir, disp);
        }
        prof_.Stop("CoarsenOperator.ApplyOp", Ns_c);

        k_loop_Ns_c iSum[k] = Zero();

        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          prof_.Start("CoarsenOperator.AccumInnerContrib");
          k_loop_Ns_c mult(calcTmp[k], MphiSplit[p * Ns_c + k], innerSites[p]);
          k_loop_Ns_c iSum[k] = iSum[k] + calcTmp[k];
          // k_loop_Ns_c iSum[k] = iSum[k] + where(TensorRemove(innerSites[p]) == vComplex(1), MphiSplit[p * Ns_c + k], zero);
          // k_loop_Ns_c iSum[k] = iSum[k] + where(innerSites[p] == vInteger(1), MphiSplit[p * Ns_c + k], zero);
          // k_loop_Ns_c iSum[k] = iSum[k] + where(innerSites[p] == Integer(1), MphiSplit[p * Ns_c + k], zero);
          // k_loop_Ns_c iSum[k] = iSum[k] + where(innerSites[p] == vComplex(1), MphiSplit[p * Ns_c + k], zero);
          // k_loop_Ns_c iSum[k] = iSum[k] + where(innerSites[p] == 1, MphiSplit[p * Ns_c + k], zero);
          // k_loop_Ns_c iSum[k] = iSum[k] + where(innerSites[p] == typename FineSiteScalar::vector_type(1), MphiSplit[p * Ns_c + k], zero);
          prof_.Stop("CoarsenOperator.AccumInnerContrib", Ns_c);

          prof_.Start("CoarsenOperator.ProjectToSubspace");
          assert(self_stencil == geom_.npoint - 1);
          if     (p == self_stencil) k_loop_Ns_c Projector.ProjectToSubspace(iProjSplit[k], iSum[k], iLut[p]);
          else if(disp == +1)        k_loop_Ns_c Projector.ProjectToSubspace(oProjSplit[k], MphiSplit[p * Ns_c + k], oLut[p]);
          prof_.Stop("CoarsenOperator.ProjectToSubspace", Ns_c);

          prof_.Start("CoarsenOperator.ConstructLinksPositive");
          Kernels::constructLinksSavingDirections(i, p, disp, self_stencil, Y_, iProjSplit, oProjSplit);
          prof_.Stop("CoarsenOperator.ConstructLinksPositive");
        }
      } else if(speedLevel_ == 2) { // save projects & save applications of backward links
        prof_.Start("CoarsenOperator.ApplyOp");
        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          if(disp == +1) k_loop_Ns_c LinOp.OpDir(phiSplit[k], MphiSplit[p * Ns_c + k], dir, disp);
        }

        k_loop_Ns_c mult(iSum[k], phiSplit[k], evenBlocks);
        k_loop_Ns_c LinOp.Op(iSum[k], MphiSplit_e[k]);
        k_loop_Ns_c mult(iSum[k], phiSplit[k], oddBlocks);
        k_loop_Ns_c LinOp.Op(iSum[k], MphiSplit_o[k]);

        {
          for(int k = 0; k < Ns_c; ++k) {
            auto iSum_v        = iSum[k].View();
            auto evenMask_v    = evenBlocks.View();
            auto oddMask_v     = oddBlocks.View();
            auto MphiSplit_e_v = MphiSplit_e[k].View();
            auto MphiSplit_o_v = MphiSplit_o[k].View();
            accelerator_for(sf, FineGrid->oSites(), Simd::Nsimd(), {
              coalescedWrite(iSum_v[sf], evenMask_v(sf) * MphiSplit_e_v(sf) + oddMask_v(sf) * MphiSplit_o_v(sf));
            });
          }
        }
        prof_.Stop("CoarsenOperator.ApplyOp", Ns_c);

        for(int p = 0; p < geom_.npoint; ++p) {
          int dir  = geom_.directions[p];
          int disp = geom_.displacements[p];

          prof_.Start("CoarsenOperator.ProjectToSubspace");
          assert(self_stencil == geom_.npoint - 1);
          if(p == self_stencil) k_loop_Ns_c Projector.ProjectToSubspace(iProjSplit[k], iSum[k], iLut[p]);
          else if(disp == +1)   k_loop_Ns_c Projector.ProjectToSubspace(oProjSplit[k], MphiSplit[p * Ns_c + k], oLut[p]);
          prof_.Stop("CoarsenOperator.ProjectToSubspace", Ns_c);

          prof_.Start("CoarsenOperator.ConstructLinksPositive");
          Kernels::constructLinksSavingDirections(i, p, disp, self_stencil, Y_, iProjSplit, oProjSplit);
          prof_.Stop("CoarsenOperator.ConstructLinksPositive");
        }
      }
    }

#undef k_loop_Ns_c

    if(speedLevel_ > 0) {
      prof_.Start("CoarsenOperator.ConstructLinksNegative");
      Kernels::shiftLinks(geom_, Y_);
      prof_.Stop("CoarsenOperator.ConstructLinksNegative");
    }

    InvertSelfStencilPoint();
    FillHalfCbs();
    prof_.Stop("CoarsenOperator.Total");
  }

  void MdirAll(const FermionField& in, std::vector<FermionField>& out) {}

private:

  void MooeeInternal(FermionField const& in, FermionField& out, int dag, int inv) {
    out.Checkerboard() = in.Checkerboard();
    assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);

    // Implementation follows clover
    LinkField* SelfStencil = nullptr;
    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd)
        SelfStencil = (inv) ? &selfStencilLinkInvOdd_ : &YOdd_[geom_.SelfStencilPoint()];
      else
        SelfStencil = (inv) ? &selfStencilLinkInvEven_ : &YEven_[geom_.SelfStencilPoint()];
    } else {
      SelfStencil = (inv) ? &selfStencilLinkInv_ : &Y_[geom_.SelfStencilPoint()];
    }
    assert(SelfStencil != nullptr);

    // TODO: is this 100% correct for all cases (dwf, clover, hermitian, ...)?
    if(dag)
      out = adj(*SelfStencil) * in;
    else
      out = *SelfStencil * in;
  }

  void InvertSelfStencilPoint() {
    prof_.Start("CoarsenOperator.InvertSelfStencilPoint");
    int localVolume = Grid()->lSites();
    int size        = Ns_c * Nc_c;

    using scalar_object = typename SiteLinkField::scalar_object;

    thread_for(site, localVolume, { // NOTE: Not able to bring this to GPU because of Eigen
      Eigen::MatrixXcd EigenSelfStencil    = Eigen::MatrixXcd::Zero(size, size);
      Eigen::MatrixXcd EigenInvSelfStencil = Eigen::MatrixXcd::Zero(size, size);

      scalar_object SelfStencilLink    = Zero();
      scalar_object InvSelfStencilLink = Zero();

      Coordinate lcoor;

      Grid()->LocalIndexToLocalCoor(site, lcoor);
      EigenSelfStencil = Eigen::MatrixXcd::Zero(size, size);
      peekLocalSite(SelfStencilLink, Y_[geom_.SelfStencilPoint()], lcoor);
      InvSelfStencilLink = Zero();

      Kernels::writeLinkToEigenMatrix(SelfStencilLink, EigenSelfStencil);
      EigenInvSelfStencil = EigenSelfStencil.inverse();
      Kernels::writeEigenMatrixToLink(EigenInvSelfStencil, InvSelfStencilLink);

      pokeLocalSite(InvSelfStencilLink, selfStencilLinkInv_, lcoor);
    });
    prof_.Stop("CoarsenOperator.InvertSelfStencilPoint");
  }

  void FillHalfCbs() {
    prof_.Start("CoarsenOperator.FillHalfCbs");
    for(int p = 0; p < geom_.npoint; ++p) {
      pickCheckerboard(Even, YEven_[p], Y_[p]);
      pickCheckerboard(Odd, YOdd_[p], Y_[p]);
    }
    pickCheckerboard(Even, selfStencilLinkInvEven_, selfStencilLinkInv_);
    pickCheckerboard(Odd, selfStencilLinkInvOdd_, selfStencilLinkInv_);
    prof_.Stop("CoarsenOperator.FillHalfCbs");
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
