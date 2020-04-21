/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/GeometryRework.h

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

class Geometry {

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:
  int              dimsToActOn;

public:
  int              ndimension;
  int              npoint;
  bool             directionsFasterIndex;
  std::vector<int> directions;
  std::vector<int> displacements;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:
  Geometry(int ndim, bool directionsFaster = false)
    : dimsToActOn((ndim >= 5) ? 4 : ndim) // coarse stencil acts on 4d lattice at most
    , ndimension(ndim)
    , npoint(2*dimsToActOn+1)
    , directionsFasterIndex(directionsFaster)
    , directions(npoint, 0)
    , displacements(npoint, 0) {
    fill();
    report();
  }

  int PointFromDirDisp(int dir, int disp) const {
    assert(disp == -1 || disp == 0 || disp == 1);
    if(ndimension == 4) assert(0 <= dir && dir < dimsToActOn);
    else if(ndimension == 5) assert(0 <= dir && dir <= dimsToActOn);

    // directions faster index = new indexing
    // 4d (base = 0):
    // dir   0  1  2  3  0  1  2  3  0
    // disp +1 +1 +1 +1 -1 -1 -1 -1  0
    // point 0  1  2  3  4  5  6  7  8
    // 5d (base = 1):
    // dir   1  2  3  4  1  2  3  4  0
    // disp +1 +1 +1 +1 -1 -1 -1 -1  0
    // point 0  1  2  3  4  5  6  7  8

    // displacements faster index = old indexing
    // 4d (base = 0):
    // dir   0  0  1  1  2  2  3  3  0
    // disp +1 -1 +1 -1 +1 -1 +1 -1  0
    // point 0  1  2  3  4  5  6  7  8
    // 5d (base = 1):
    // dir   1  1  2  2  3  3  4  4  0
    // disp +1 -1 +1 -1 +1 -1 +1 -1  0
    // point 0  1  2  3  4  5  6  7  8

    if(dir == 0 and disp == 0)
      return 2 * dimsToActOn;
    else {
      if(directionsFasterIndex)
        return (1 - disp) / 2 * dimsToActOn + dir - base();
      else
        return (dimsToActOn * (dir - base()) + 1 - disp) / 2;
    }
  }

  int SelfStencilPoint() const { return npoint - 1; } // always the last in the list

  bool DirectionsFasterIndex() const { return directionsFasterIndex; }

  bool DisplacementsFasterIndex() const { return !directionsFasterIndex; }

private:
  int base() const { return (ndimension == 5) ? 1 : 0; }

  void fill() {
    for(int d = 0; d < dimsToActOn; d++) {
      if(directionsFasterIndex) { // directions faster index
        directions[d]        = d + base();
        directions[d + 4]    = d + base();
        displacements[d]     = +1;
        displacements[d + 4] = -1;
      } else { // displacements faster index
        directions[2 * d]        = d + base();
        directions[2 * d + 1]    = d + base();
        displacements[2 * d]     = +1;
        displacements[2 * d + 1] = -1;
      }
    }
    directions[npoint - 1]    = 0;
    displacements[npoint - 1] = 0;
  }

  void report() const {
    std::cout << GridLogMessage << "directions    :";
    for(int d = 0; d < npoint; d++) std::cout << directions[d] << " ";
    std::cout << std::endl;
    std::cout << GridLogMessage << "displacements :";
    for(int d = 0; d < npoint; d++) std::cout << displacements[d] << " ";
    std::cout << std::endl;
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
