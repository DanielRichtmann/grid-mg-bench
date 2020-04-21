/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_geom.cc

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

#include <Grid/Grid.h>
#include <Multigrid.h>

using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  using Geometry = Grid::Rework::Geometry;

  std::cout << GridLogMessage <<"==================================="<<  std::endl;
  std::cout << GridLogMessage <<"Testing class Geometry in 4 dimensions "<<std::endl;
  std::cout << GridLogMessage <<"==================================="<<  std::endl;

  for(bool directionsFasterIndex : {false, true}) {
    Geometry geom4d(4, directionsFasterIndex);

    for(int p = 0; p < geom4d.npoint; ++p) {
      int dir  = geom4d.directions[p];
      int disp = geom4d.displacements[p];
      assert(p == geom4d.PointFromDirDisp(dir, disp));
    }

    std::cout << GridLogMessage << "directionsFasterIndex = " << directionsFasterIndex << " -- OK ! " << std::endl;
  }

  std::cout << GridLogMessage << "OK ! " << std::endl;

  std::cout << GridLogMessage <<"==================================="<<  std::endl;
  std::cout << GridLogMessage <<"Testing class Geometry in 5 dimensions "<<std::endl;
  std::cout << GridLogMessage <<"==================================="<<  std::endl;

  for(bool directionsFasterIndex : {false, true}) {
    Geometry geom5d(5, directionsFasterIndex);

    for(int p = 0; p < geom5d.npoint; ++p) {
      int dir  = geom5d.directions[p];
      int disp = geom5d.displacements[p];
      assert(p == geom5d.PointFromDirDisp(dir, disp));
    }

    std::cout << GridLogMessage << "directionsFasterIndex = " << directionsFasterIndex << " -- OK ! " << std::endl;
  }

  std::cout << GridLogMessage <<"OK ! "<<std::endl;

  Grid_finalize();
}
