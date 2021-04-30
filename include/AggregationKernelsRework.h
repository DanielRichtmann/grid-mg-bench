/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/AggregationKernelsRework.h

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
class AggregationKernels : public CoarseningPolicy {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
  INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);
  using CoarseningLookupTable = Grid::Rework::CoarseningLookupTable;

  /////////////////////////////////////////////
  // Kernel functions
  /////////////////////////////////////////////

  static void aggregateProject(FermionField&                coarseVec,
                               FineFermionField const&      fineVec,
                               IntergridOperator const&     projector,
                               CoarseningLookupTable const& lut) {
    GridBase* fineGrid   = fineVec.Grid();
    GridBase* coarseGrid = coarseVec.Grid();
    int       nDimension = coarseGrid->_ndimension;

    // checks
    assert(projector.size() == Nc_c);
    subdivides(coarseGrid, fineGrid);
    for(auto const& elem : projector) conformable(elem, fineVec);

    Coordinate block_r(nDimension);
    for(int d = 0; d < nDimension; ++d) {
      block_r[d] = fineGrid->_rdimensions[d] / coarseGrid->_rdimensions[d];
      assert(block_r[d] * coarseGrid->_rdimensions[d] == fineGrid->_rdimensions[d]);
    }

    coarseVec = Zero();

    auto fineVec_v   = fineVec.View();
    auto coarseVec_v = coarseVec.View();
    thread_for(sf, fineGrid->oSites(), {
      int        sc;
      Coordinate coor_c(nDimension);
      Coordinate coor_f(nDimension);

      Lexicographic::CoorFromIndex(coor_f, sf, fineGrid->_rdimensions);
      for(int d = 0; d < nDimension; ++d) coor_c[d] = coor_f[d] / block_r[d];
      Lexicographic::IndexFromCoor(coor_c, sc, coarseGrid->_rdimensions);

      auto siteInLut = std::find(lut()[sc].begin(), lut()[sc].end(), sf) != lut()[sc].end();
      if(!siteInLut) continue;

      thread_critical {
        for(int i = 0; i < Nc_c; ++i) {
          auto projector_v = projector[i].View();
          for(int s = 0; s < Ns_f; ++s)
            coarseVec_v[sc]()(s / Ns_b)(i) =
              coarseVec_v[sc]()(s / Ns_b)(i) +
              TensorRemove(innerProduct(projector_v[sf]()(s), fineVec_v[sf]()(s)));
        }
      }
    });
  }

  static void aggregateProjectFast(FermionField&                coarseVec,
                                   FineFermionField const&      fineVec,
                                   IntergridOperator const&     projector,
                                   CoarseningLookupTable const& lut) {
    GridBase* fineGrid   = fineVec.Grid();
    GridBase* coarseGrid = coarseVec.Grid();

    auto ndimU = projector[0].Grid()->_ndimension;
    auto ndimF = coarseGrid->_ndimension;

    assert(fineGrid->_ndimension == ndimF);
    assert(ndimF == ndimU || ndimF == ndimU+1);
    assert(projector.size() == Nc_c);

    int LLs = 1;
    if(ndimF == ndimU) {
      assert(lut.gridPointersMatch(coarseGrid, fineGrid));
      for(auto const& elem : projector) conformable(elem, fineVec);
      LLs = 1;
    } else if(ndimF == ndimU+1) {
      assert(coarseGrid->_fdimensions[0] == coarseGrid->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
      assert(fineGrid->_fdimensions[0] == fineGrid->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
      assert(coarseGrid->_rdimensions[0] == fineGrid->_rdimensions[0]); // same extent in 5th dimension
      LLs = coarseGrid->_rdimensions[0];
    }

    coarseVec = Zero(); // runs on CPU

    typedef CoarseningLookupTable::size_type size_type;

    auto fineVec_v   = fineVec.View();
    auto coarseVec_v = coarseVec.View();
    auto lut_v       = lut.View();
    auto sizes_v     = lut.Sizes();

    vectorViewPointerOpen(projector_v, projector_p, projector, AcceleratorRead);

    accelerator_for(scF, coarseGrid->oSites(), Simd::Nsimd(), {
      auto scU = scF/LLs;
      auto s5  = scF%LLs;

      auto coarseVec_t = coarseVec_v(scF);
      // decltype(coalescedRead(coarseVec_v[sc])) coarseVec_t = Zero(); // could use this instead and comment line with Zero() above

      for(size_type i = 0; i < sizes_v[scU]; ++i) {
        auto sfU = lut_v[scU][i];
        auto sfF = sfU*LLs + s5;

        auto fineVec_t = fineVec_v(sfF);
        for(int c = 0; c < Nc_c; ++c) {
          auto projector_t = coalescedRead(projector_p[c][sfU]);
          for(int s = 0; s < Ns_f; ++s) {
            coarseVec_t()(s / Ns_b)(c) =
              coarseVec_t()(s / Ns_b)(c) +
              TensorRemove(innerProduct(projector_t()(s), fineVec_t()(s)));
          }
        }
      }
      coalescedWrite(coarseVec_v[scF], coarseVec_t);
    });
    vectorViewPointerClose(projector_v, projector_p);
  }

  static void aggregatePromote(FermionField const&          coarseVec,
                               FineFermionField&            fineVec,
                               IntergridOperator const&     projector,
                               CoarseningLookupTable const& lut) {
    GridBase* fineGrid   = fineVec.Grid();
    GridBase* coarseGrid = coarseVec.Grid();
    int       nDimension = coarseGrid->_ndimension;

    // checks
    assert(projector.size() == Nc_c);
    subdivides(coarseGrid, fineGrid);
    for(auto const& elem : projector) conformable(elem, fineVec);

    Coordinate block_r(nDimension);
    for(int d = 0; d < nDimension; ++d) {
      block_r[d] = fineGrid->_rdimensions[d] / coarseGrid->_rdimensions[d];
      assert(block_r[d] * coarseGrid->_rdimensions[d] == fineGrid->_rdimensions[d]);
    }

    auto fineVec_v   = fineVec.View();
    auto coarseVec_v = coarseVec.View();
    thread_for(sf, fineGrid->oSites(), {
      int        sc;
      Coordinate coor_c(nDimension);
      Coordinate coor_f(nDimension);

      Lexicographic::CoorFromIndex(coor_f, sf, fineGrid->_rdimensions);
      for(int d = 0; d < nDimension; ++d) coor_c[d] = coor_f[d] / block_r[d];
      Lexicographic::IndexFromCoor(coor_c, sc, coarseGrid->_rdimensions);

      iScalar<Simd> coarseTmp;
      for(int i = 0; i < Nc_c; ++i) {
        auto projector_v = projector[i].View();
        if(i == 0) {
          for(int s = 0; s < Ns_f; ++s) {
            coarseTmp          = coarseVec_v[sc]()(s / Ns_b)(i);
            fineVec_v[sf]()(s) = coarseTmp * projector_v[sf]()(s);
          }
        } else {
          for(int s = 0; s < Ns_f; ++s) {
            coarseTmp          = coarseVec_v[sc]()(s / Ns_b)(i);
            fineVec_v[sf]()(s) = fineVec_v[sf]()(s) + coarseTmp * projector_v[sf]()(s);
          }
        }
      }
    });
  }

  static void aggregatePromoteFast(FermionField const&          coarseVec,
                                   FineFermionField&            fineVec,
                                   IntergridOperator const&     projector,
                                   CoarseningLookupTable const& lut) {
    GridBase* fineGrid   = fineVec.Grid();
    GridBase* coarseGrid = coarseVec.Grid();

    auto ndimU = projector[0].Grid()->_ndimension;
    auto ndimF = coarseGrid->_ndimension;

    assert(fineGrid->_ndimension == ndimF);
    assert(ndimF == ndimU || ndimF == ndimU+1);
    assert(projector.size() == Nc_c);

    int LLs = 1;
    if(ndimF == ndimU) {
      assert(lut.gridPointersMatch(coarseGrid, fineGrid));
      for(auto const& elem : projector) conformable(elem, fineVec);
      LLs = 1;
    } else if(ndimF == ndimU+1) {
      assert(coarseGrid->_fdimensions[0] == coarseGrid->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
      assert(fineGrid->_fdimensions[0] == fineGrid->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
      assert(coarseGrid->_rdimensions[0] == fineGrid->_rdimensions[0]); // same extent in 5th dimension
      LLs = coarseGrid->_rdimensions[0];
    }

    typedef CoarseningLookupTable::size_type size_type;

    auto fineVec_v   = fineVec.View();
    auto coarseVec_v = coarseVec.View();
    auto rlut_v      = lut.ReverseView();

    vectorViewPointerOpen(projector_v, projector_p, projector, AcceleratorRead);

    accelerator_for(sfF, fineGrid->oSites(), Simd::Nsimd(), {
      auto sfU = sfF/LLs;
      auto s5  = sfF%LLs;
      auto scU = rlut_v[sfU];
      auto scF = scU*LLs + s5;

      auto fineVec_t   = fineVec_v(sfF);
      auto coarseVec_t = coarseVec_v(scF);

      iScalar<typename decltype(coarseVec_t)::vector_type> tmp;
      for(int i = 0; i < Nc_c; ++i) {
        auto projector_t = coalescedRead(projector_p[i][sfU]);
        for(int s = 0; s < Ns_f; ++s) {
          tmp() = coarseVec_t()(s / Ns_b)(i);
          if(i == 0)
            fineVec_t()(s) = tmp * projector_t()(s);
          else
            fineVec_t()(s) = fineVec_t()(s) + tmp * projector_t()(s);
        }
      }
      coalescedWrite(fineVec_v[sfF], fineVec_t);
    });
    vectorViewPointerClose(projector_v, projector_p);
  }

  static void aggregateOrthogonalise(ScalarField&                 innerProd,
                                     IntergridOperator&           projector,
                                     CoarseningLookupTable const& lut) {
    GridBase* fineGrid   = projector[0].Grid();
    GridBase* coarseGrid = innerProd.Grid();

    subdivides(coarseGrid, fineGrid);
    assert(projector.size() == Nc_c);
    for(int i = 0; i < Nc_c; ++i) conformable(projector[i].Grid(), fineGrid);

    SpinVectorField alpha(coarseGrid);
    SpinVectorField norm(coarseGrid);

    typedef CoarseningLookupTable::size_type size_type;

    auto alpha_v = alpha.View();
    auto norm_v  = norm.View();

    auto lut_v   = lut.View();
    auto sizes_v = lut.Sizes();

    vectorViewPointerOpen(projector_v, projector_p, projector, AcceleratorWrite);

    // Kernel fusion
    accelerator_for(sc, coarseGrid->oSites(), Simd::Nsimd(), {
      auto alpha_t = alpha_v(sc);
      auto norm_t  = norm_v(sc);

      for(int v = 0; v < Nc_c; ++v) {
        for(int u = 0; u < v; ++u) {
          for(int k = 0; k < Ns_c; ++k) alpha_t()(k) = Zero();

          // alpha = <proj[u], proj[v]>
          for(size_type i = 0; i < sizes_v[sc]; ++i) {
            auto sf    = lut_v[sc][i];
            auto p_u_t = coalescedRead(projector_p[u][sf]);
            auto p_v_t = coalescedRead(projector_p[v][sf]);
            for(int s = 0; s < Ns_f; ++s)
              alpha_t()(s / Ns_b) = alpha_t()(s / Ns_b) + innerProduct((p_u_t()(s)), p_v_t()(s));
          }

          // proj[v] -= alpha * proj[u]
          for(size_type i = 0; i < sizes_v[sc]; ++i) {
            auto sf    = lut_v[sc][i];
            auto p_u_t = coalescedRead(projector_p[u][sf]);
            auto p_v_t = coalescedRead(projector_p[v][sf]);
            for(int s = 0; s < Ns_f; ++s)
              p_v_t()(s) = p_v_t()(s) - alpha_t()(s / Ns_b) * p_u_t()(s);
            coalescedWrite(projector_p[v][sf], p_v_t);
          }
        }

        for(int k = 0; k < Ns_c; ++k) norm_t()(k) = Zero();

        // norm = <proj[v], proj[v]>
        for(size_type i = 0; i < sizes_v[sc]; ++i) {
          auto sf    = lut_v[sc][i];
          auto p_v_t = coalescedRead(projector_p[v][sf]);
          for(int s = 0; s < Ns_f; ++s)
            norm_t()(s / Ns_b) = norm_t()(s / Ns_b) + innerProduct(p_v_t()(s), p_v_t()(s));
        }

        for(int k = 0; k < Ns_c; ++k) norm_t()(k) = pow(norm_t()(k), -0.5);

        // proj[v] = 1/norm * proj[v]
        for(size_type i = 0; i < sizes_v[sc]; ++i) {
          auto sf    = lut_v[sc][i];
          auto p_v_t = coalescedRead(projector_p[v][sf]);
          for(int s = 0; s < Ns_f; ++s)
            p_v_t()(s) = norm_t()(s / Ns_b) * p_v_t()(s);
          coalescedWrite(projector_p[v][sf], p_v_t);
        }
      }
    });
    vectorViewPointerClose(projector_v, projector_p);
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
