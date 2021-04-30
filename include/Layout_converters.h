/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Layout_converters.h

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
NAMESPACE_BEGIN(LayoutConverters);

template<class Simd, class CComplex, int nbasis, int ncs>
void convertLayout(Lattice<iScalar<iMatrix<iMatrix<Simd, nbasis / ncs>, ncs>>> const& in,
                   Lattice<iMatrix<CComplex, nbasis>>&                                out) {
  conformable(in.Grid(), out.Grid());
  GridBase* grid   = out.Grid();
  int       Nbasis = nbasis;
  int       Ns_c   = ncs;
  int       Nc_c   = Nbasis / Ns_c;

  typedef iMatrix<CComplex, nbasis> InnerTypeOut;

  autoView(out_v, out, AcceleratorWrite);
  autoView(in_v , in, AcceleratorRead);
  accelerator_for(ss, grid->oSites(), Simd::Nsimd(), {
    decltype(coalescedRead(InnerTypeOut())) out_t;
    auto in_t = in_v(ss);
    for(int n1 = 0; n1 < Nbasis; n1++) {
      for(int n2 = 0; n2 < Nbasis; n2++) {
        int spinIndex1   = n1 / Nc_c;
        int spinIndex2   = n2 / Nc_c;
        int colourIndex1 = n1 % Nc_c;
        int colourIndex2 = n2 % Nc_c;
        out_t(n1, n2)()()() = in_t()(spinIndex1, spinIndex2)(colourIndex1, colourIndex2);
      }
    }
    coalescedWrite(out_v[ss], out_t);
  });
}

template<class Simd, class CComplex, int nbasis, int ncs>
void convertLayout(Lattice<iMatrix<CComplex, nbasis>>&                                in,
                   Lattice<iScalar<iMatrix<iMatrix<Simd, nbasis / ncs>, ncs>>> const& out) {
  conformable(in.Grid(), out.Grid());
  GridBase* grid   = out.Grid();
  int       Nbasis = nbasis;
  int       Ns_c   = ncs;
  int       Nc_c   = Nbasis / Ns_c;

  typedef iScalar<iMatrix<iMatrix<Simd, nbasis / ncs>, ncs>> InnerTypeOut;

  autoView(out_v, out, AcceleratorWrite);
  autoView(in_v , in, AcceleratorRead);
  accelerator_for(ss, grid->oSites(), Simd::Nsimd(), {
    decltype(coalescedRead(InnerTypeOut())) out_t;
    auto in_t = in_v(ss);
    for(int n1 = 0; n1 < Nbasis; n1++) {
      for(int n2 = 0; n2 < Nbasis; n2++) {
        int spinIndex1   = n1 / Nc_c;
        int spinIndex2   = n2 / Nc_c;
        int colourIndex1 = n1 % Nc_c;
        int colourIndex2 = n2 % Nc_c;
        out_t()(spinIndex1, spinIndex2)(colourIndex1, colourIndex2) = in_t(n1, n2)()()();
      }
    }
    coalescedWrite(out_v[ss], out_t);
  });
}

template<class Simd, class CComplex, int nbasis, int ncs>
void convertLayout(Lattice<iScalar<iVector<iVector<Simd, nbasis / ncs>, ncs>>> const& in,
                   Lattice<iVector<CComplex, nbasis>>&                                out) {
  conformable(in.Grid(), out.Grid());
  GridBase* grid   = out.Grid();
  int       Nbasis = nbasis;
  int       Ns_c   = ncs;
  int       Nc_c   = Nbasis / Ns_c;

  typedef iVector<CComplex, nbasis> InnerTypeOut;

  autoView(out_v, out, AcceleratorWrite);
  autoView(in_v , in, AcceleratorRead);
  accelerator_for(ss, grid->oSites(), Simd::Nsimd(), {
    decltype(coalescedRead(InnerTypeOut())) out_t;
    auto in_t = in_v(ss);
    for(int n = 0; n < Nbasis; n++) {
      int spinIndex   = n / Nc_c;
      int colourIndex = n % Nc_c;
      out_t(n)()()() = in_t()(spinIndex)(colourIndex);
    }
    coalescedWrite(out_v[ss], out_t);
  });
}

template<class Simd, class CComplex, int nbasis, int ncs>
void convertLayout(Lattice<iVector<CComplex, nbasis>> const&                    in,
                   Lattice<iScalar<iVector<iVector<Simd, nbasis / ncs>, ncs>>>& out) {
  conformable(in.Grid(), out.Grid());
  GridBase* grid   = in.Grid();
  int       Nbasis = nbasis;
  int       Ns_c   = ncs;
  int       Nc_c   = Nbasis / Ns_c;

  typedef iScalar<iVector<iVector<Simd, nbasis / ncs>, ncs>> InnerTypeOut;

  autoView(out_v, out, AcceleratorWrite);
  autoView(in_v , in, AcceleratorRead);
  accelerator_for(ss, grid->oSites(), Simd::Nsimd(), {
    decltype(coalescedRead(InnerTypeOut())) out_t;
    auto in_t = in_v(ss);
    for(int n = 0; n < Nbasis; n++) {
      int spinIndex   = n / Nc_c;
      int colourIndex = n % Nc_c;
      out_t()(spinIndex)(colourIndex) = in_t(n)()()();
    }
    coalescedWrite(out_v[ss], out_t);
  });
}

NAMESPACE_END(LayoutConverters);
NAMESPACE_END(Grid);
