/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/LevelInfo.h

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

class LevelInfo {
public:
  std::vector<std::vector<int>>       Seeds;
  std::vector<GridCartesian*>         Grids;
  std::vector<GridRedBlackCartesian*> RBGrids;
  std::vector<GridParallelRNG>        PRNGs;

  LevelInfo(GridCartesian* FineGrid, MultiGridParams const& mgParams) {
    auto nCoarseLevels = mgParams.blockSizes.size();

    assert(nCoarseLevels == mgParams.nLevels - 1);

    // set up values for finest grid
    Grids.push_back(FineGrid);
    RBGrids.push_back((FineGrid->Nd() == 4) ? SpaceTimeGrid::makeFourDimRedBlackGrid(Grids.back())
                                            : SpaceTimeGrid::makeFiveDimRedBlackGrid(1, Grids.back()));
    Seeds.push_back((FineGrid->Nd() == 4) ? std::vector<int>{1, 2, 3, 4} : std::vector<int>{5, 6, 7, 8});
    PRNGs.push_back(GridParallelRNG(Grids.back()));
    PRNGs.back().SeedFixedIntegers(Seeds.back());

    // set up values for coarser grids
    for(int level = 1; level < mgParams.nLevels; ++level) {
      auto Nd               = Grids[level - 1]->Nd();
      auto fullDimensions4d = extract4dData(Grids[level - 1]->FullDimensions());

      assert(Nd == 4 || Nd == 5);
      assert(fullDimensions4d.size() == mgParams.blockSizes[level - 1].size());

      Seeds.push_back(std::vector<int>(4));

      for(int d = 0; d < fullDimensions4d.size(); ++d) {
        fullDimensions4d[d] /= mgParams.blockSizes[level - 1][d];
        Seeds[level][d] = (level)*Nd + d + 1;
      }

      auto           simdLayout4d = extract4dData(Grids[level - 1]->_simd_layout);
      auto           mpiLayout4d  = extract4dData(Grids[level - 1]->_processors);
      GridCartesian* tmpGrid4d = SpaceTimeGrid::makeFourDimGrid(fullDimensions4d, simdLayout4d, mpiLayout4d);
      if(Nd == 4) {
        Grids.push_back(tmpGrid4d);
        RBGrids.push_back(SpaceTimeGrid::makeFourDimRedBlackGrid(tmpGrid4d));
      } else {
        Grids.push_back(SpaceTimeGrid::makeFiveDimGrid(1, tmpGrid4d));
        RBGrids.push_back(SpaceTimeGrid::makeFiveDimRedBlackGrid(1, tmpGrid4d));
      }

      PRNGs.push_back(GridParallelRNG(Grids[level]));
      PRNGs[level].SeedFixedIntegers(Seeds[level]);
    }

    std::cout << GridLogMessage << "Constructed " << mgParams.nLevels << " levels" << std::endl;

    for(int level = 0; level < mgParams.nLevels; ++level) {
      std::cout << GridLogMessage << "level = " << level << ":" << std::endl;
      Grids[level]->show_decomposition();
      RBGrids[level]->show_decomposition();
    }
  }

private:
  Coordinate extract4dData(Coordinate const& data) {
    assert(data.size() == 4 || data.size() == 5);
    return Coordinate(std::vector<int>(data.begin() + (data.size() - 4), data.end()));
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
