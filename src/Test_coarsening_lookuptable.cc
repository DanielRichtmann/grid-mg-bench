/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests_multigrid/Test_coarsening_lookuptable.cc

    Copyright (C) 2015 - 2020

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>

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

#include <Grid/Grid.h>
#include <tests/multigrid/Multigrid.h>
#include <benchmarks/Benchmark_helpers.h>

using namespace Grid;
using namespace Grid::BenchmarkHelpers;
using namespace Grid::Rework;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate blockSize = readFromCommandLineCoordinate(&argc, &argv, "--blocksize", Coordinate({2, 2, 2, 2}));

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  Coordinate clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian*         FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian*         CGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());

  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "CGrid:" << std::endl; CGrid->show_decomposition();

  CoarseningLookupTable lut(CGrid, FGrid);

  // this works!
  // Vector<index_type const*> tmp;
  // for(index_type sc = 0; sc < lut().size(); ++sc) tmp.push_back(&lut()[sc][0]);
  // auto** lut_p = &tmp[0];

  /////////////////////////////////////////////////////////////////////////////
  //                              Begin of tests                             //
  /////////////////////////////////////////////////////////////////////////////

  typedef CoarseningLookupTable::index_type index_type;
  typedef CoarseningLookupTable::size_type size_type;

  auto lut_v   = lut.View();
  auto sizes_v = lut.Sizes();
  auto rlut_v  = lut.ReverseView();

  for(index_type sc = 0; sc < lut().size(); ++sc) {
    for(index_type i = 0; i < lut()[sc].size(); ++i) {
      auto sf_original  = lut()[sc][i];
      auto sf_ptr       = lut_v[sc][i];
      auto sc_reverse   = rlut_v[sf_original];
      auto diff_ptr     = sf_original - sf_ptr;
      auto diff_reverse = sc - sc_reverse;
      assert(diff_ptr == 0);
      assert(diff_reverse == 0);
      // clang-format off
      std::cout << GridLogDebug
                << "sc = "             << sc
                << ", i = "            << i
                << ", sf_original = "  << sf_original
                << ", sf_ptr = "       << sf_ptr
                << ", diff_ptr = "     << diff_ptr
                << ", sc_reverse = "   << sc_reverse
                << ", diff_reverse = " << diff_reverse
                << std::endl;
      // clang-format on
    }
  }

  accelerator_for(sc, CGrid->oSites(), vComplex::Nsimd(), {
    for(size_type i = 0; i < sizes_v[sc]; ++i) {
      auto sf = lut_v[sc][i];
      printf("GPU_ACCESSIBILITY_NORMAL: sc = %lu, i = %lu, sf = %lu\n", sc, i, sf);
    }
  });

  accelerator_for(sf, FGrid->oSites(), vComplex::Nsimd(), {
    auto sc = rlut_v[sf];
    printf("GPU_ACCESSIBILITY_REVERSE: sf = %lu, sc = %lu\n", sf, sc);
  });

  Grid_finalize();
}
