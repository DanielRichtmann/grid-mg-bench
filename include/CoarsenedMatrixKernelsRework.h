/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/CoarsenedMatrixKernelsRework.h

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
class CoarsenedMatrixKernels : public CoarseningPolicy {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
  INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);
  using Geometry = Grid::Rework::Geometry;

  /////////////////////////////////////////////
  // Kernel functions
  /////////////////////////////////////////////

  static void extractSpinComponents(FineFermionField const& in, std::vector<FineFermionField>& out) {
    assert(out.size() == Ns_c);

    if(Ns_c == 1) {
      out[0] = in;
      return;
    }

    for(int k = 0; k < Ns_c; ++k) conformable(in.Grid(), out[k].Grid());

    for(int k = 0; k < Ns_c; ++k) out[k] = Zero();

    auto  out_v_c   = getViewContainer(out);
    auto* out_v_c_p = &out_v_c[0];

    auto in_v = in.View();

    if(in.Grid()->_ndimension == 0) { // actually there is a 5 here, 0 just for testing
      int      Ls    = in.Grid()->_rdimensions[0];
      uint64_t nLoop = in.Grid()->oSites() / Ls;
      accelerator_for(sss, nLoop, Simd::Nsimd(), {
        uint64_t ss = sss * Ls;
        for(int s5 = 0; s5 < Ls; ++s5) {
          int s5r = Ls - 1 - s5;
          auto tmp = in_v(ss + s5);
          for(int s = 0; s < Ns_f; ++s)
            coalescedWrite(out_v_c_p[s / Ns_b][ss + s5r]()(s), tmp()(s));
        }
      });
    } else {
      accelerator_for(ss, in.Grid()->oSites(), Simd::Nsimd(), {
        auto tmp = in_v(ss);
        for(int s = 0; s < Ns_f; ++s)
          coalescedWrite(out_v_c_p[s / Ns_b][ss]()(s), tmp()(s));
      });
    }
  }

  static void constructLinksFull(int                              i,
                                 int                              p,
                                 int                              disp,
                                 int                              self_stencil,
                                 std::vector<LinkField>&          Y,
                                 std::vector<FermionField> const& iProjSplit,
                                 std::vector<FermionField> const& oProjSplit) {
    auto Y_p_v    = Y[p].View();
    auto Y_self_v = Y[self_stencil].View();

    auto iProjSplit_v_c = getViewContainer(iProjSplit);
    auto oProjSplit_v_c = getViewContainer(oProjSplit);

    auto* iProjSplit_v_c_p = &iProjSplit_v_c[0];
    auto* oProjSplit_v_c_p = &oProjSplit_v_c[0];

    accelerator_for(ss, Y[p].Grid()->oSites(), Simd::Nsimd(), {
      auto Y_p_t    = Y_p_v(ss);
      auto Y_self_t = Y_self_v(ss);
      for(int j = 0; j < Nc_c; ++j) {
        if(disp != 0) {
          for(int k = 0; k < Ns_c; ++k) {
            auto oProjSplit_t = coalescedRead(oProjSplit_v_c_p[k][ss]);
            for(int l = 0; l < Ns_c; ++l)
              Y_p_t()(l, k)(j, i) = oProjSplit_t()(l)(j);
          }
        }

        for(int k = 0; k < Ns_c; ++k) {
          auto iProjSplit_t = coalescedRead(iProjSplit_v_c_p[k][ss]);
          for(int l = 0; l < Ns_c; ++l)
            Y_self_t()(l, k)(j, i) = Y_self_t()(l, k)(j, i) + iProjSplit_t()(l)(j);
        }
      }
      coalescedWrite(Y_p_v[ss], Y_p_t);
      coalescedWrite(Y_self_v[ss], Y_self_t);
    });
  }

  static void constructLinkField(int i, LinkField& Yp, std::vector<FermionField> const& projSplit) {
    auto  Yp_v            = Yp.View();
    auto  projSplit_v_c   = getViewContainer(projSplit);
    auto* projSplit_v_c_p = &projSplit_v_c[0];

    accelerator_for(ss, Yp.Grid()->oSites(), Simd::Nsimd(), {
      for(int j = 0; j < Nc_c; ++j) {
        for(int k = 0; k < Ns_c; ++k) {
          auto projSplit_t = coalescedRead(projSplit_v_c_p[k][ss]);
          for(int l = 0; l < Ns_c; ++l)
            coalescedWrite(Yp_v[ss]()(l, k)(j, i), projSplit_t()(l)(j));
        }
      }
    });
  }

  static void shiftLinks(Geometry& geom, std::vector<LinkField>& Y, int dispHave) {
    assert(dispHave == +1 || dispHave == -1);
    // Relation between forward and backward link matrices taken from M. Rottmann's PHD thesis:
    // D_{A_{q,\kappa}, A_{p,\tau}} = - D^\dag_{A_{p,\tau}, A_{q,\kappa}}
    int dispWant = dispHave * -1;
    for(int p = 0; p < geom.npoint; ++p) {
      int dir  = geom.directions[p];
      int disp = geom.displacements[p];
      if(disp == dispHave) {
        auto tmp   = adj(Y[p]);
        auto tmp_v = tmp.View();
        accelerator_for(ss, tmp.Grid()->oSites(), Simd::Nsimd(), {
          Real factor;
          auto tmp_t = tmp_v(ss);
          for(int k = 0; k < Ns_c; ++k) {
            for(int l = 0; l < Ns_c; ++l) {
              factor        = ((k + l) % 2 == 1) ? -1. : 1.;
              tmp_t()(k, l) = factor * tmp_t()(k, l);
            }
          }
          coalescedWrite(tmp_v[ss], tmp_t);
        });
        int p_partner = geom.PointFromDirDisp(dir, dispWant);
        Y[p_partner] = Cshift(tmp, dir, dispWant);
      }
    }
  }

  static void writeLinkToEigenMatrix(typename SiteLinkField::scalar_object const& linkMat,
                                     Eigen::MatrixXcd&                            eigenMat) {
    for(int s1 = 0; s1 < Ns_c; ++s1)
      for(int s2 = 0; s2 < Ns_c; ++s2)
        for(int c1 = 0; c1 < Nc_c; ++c1)
          for(int c2 = 0; c2 < Nc_c; ++c2) {
            int n1           = s1 * Nc_c + c1;
            int n2           = s2 * Nc_c + c2;
            eigenMat(n1, n2) = linkMat()(s1, s2)(c1, c2);
          }
  }

  static void writeEigenMatrixToLink(Eigen::MatrixXcd const&                eigenMat,
                                     typename SiteLinkField::scalar_object& linkMat) {
    for(int s1 = 0; s1 < Ns_c; ++s1)
      for(int s2 = 0; s2 < Ns_c; ++s2)
        for(int c1 = 0; c1 < Nc_c; ++c1)
          for(int c2 = 0; c2 < Nc_c; ++c2) {
            int n1                    = s1 * Nc_c + c1;
            int n2                    = s2 * Nc_c + c2;
            linkMat()(s1, s2)(c1, c2) = eigenMat(n1, n2);
          }
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
