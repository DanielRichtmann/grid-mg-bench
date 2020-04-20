/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/CoarseningLookupTableRework.h

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

class CoarseningLookupTable {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  typedef uint64_t index_type;
  typedef uint64_t size_type;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:
  GridBase*                       coarse_;
  GridBase*                       fine_;
  bool                            isPopulated_;
  std::vector<Vector<index_type>> lutVec_;
  Vector<index_type const*>       lutPtr_;
  Vector<size_type>               sizes_;
  Vector<index_type>              reverseLutVec_;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:
  CoarseningLookupTable(GridBase* coarse, GridBase* fine)
    : coarse_(coarse)
    , fine_(fine)
    , isPopulated_(false)
    , lutVec_(coarse_->oSites())
    , lutPtr_(coarse_->oSites())
    , sizes_(coarse_->oSites())
    , reverseLutVec_(fine_->oSites()) {
    populate(coarse_, fine_);
  }

  CoarseningLookupTable()
    : coarse_(nullptr)
    , fine_(nullptr)
    , isPopulated_(false)
    , lutVec_()
    , lutPtr_()
    , sizes_()
    , reverseLutVec_()
  {}

  // clang-format off
  ~CoarseningLookupTable()                                       = default;
  CoarseningLookupTable(const CoarseningLookupTable&)            = delete;
  CoarseningLookupTable& operator=(CoarseningLookupTable const&) = delete;
  CoarseningLookupTable(CoarseningLookupTable&&)                 = delete;
  CoarseningLookupTable& operator=(CoarseningLookupTable&&)      = delete;

  accelerator_inline std::vector<Vector<index_type>> const& operator()()  const { return lutVec_; }     // CPU access (TODO: remove?)
  accelerator_inline index_type const* const*               View()        const { return &lutPtr_[0]; } // GPU access
  accelerator_inline size_type  const*                      Sizes()       const { return &sizes_[0]; }  // also needed for GPU access
  accelerator_inline index_type const*                      ReverseView() const { return &reverseLutVec_[0]; }
  // clang-format on

  bool isPopulated() const { return isPopulated_; }

  bool gridPointersMatch(GridBase* coarse, GridBase* fine) const {
    // NOTE: This is the same check that "conformable" does
    return (coarse == coarse_) && (fine == fine_);
  }

  void setGridPointers(GridBase* coarse, GridBase* fine) {
    coarse_ = coarse;
    fine_   = fine;
  }

  void populate(GridBase* coarse, GridBase* fine) {
    if(gridPointersMatch(coarse, fine) && isPopulated()) {
      std::cout << GridLogMessage << "No recalculation of coarsening lookup table needed. Skipping"
                << std::endl;
      return;
    }
    setGridPointers(coarse, fine);
    populate();
    std::cout << GridLogMessage << "Recalculation of coarsening lookup table finished" << std::endl;
  }

  template<typename ScalarField>
  void deleteUnneededFineSites(ScalarField const& in) {
    assert(in.Grid() == fine_);

    typename ScalarField::scalar_type zz(0);

    // TODO: Is this correct if there are simd sites within different aggregates / can this situation happen at all?
    auto in_v = in.View();
    thread_for(sc, coarse_->oSites(), { // NOTE: this won't work on gpu
      Vector<index_type> tmp;
      for(index_type i = 0; i < lutVec_[sc].size(); ++i) {
        int sf = lutVec_[sc][i];
        if(Reduce(TensorRemove(in_v(sf))) != zz) tmp.push_back(sf);
      }
      lutVec_[sc] = tmp;
      sizes_[sc]  = lutVec_[sc].size();
      lutPtr_[sc] = &lutVec_[sc][0];
    });
    // NOTE: Nothing to do with reverseLutVec_ here since
    // a lut "with holes" is not a use case for the reverse table
  }

private:
  void populate() {
    int        _ndimension = coarse_->_ndimension;
    Coordinate block_r(_ndimension);
    Coordinate coor_c(_ndimension);
    Coordinate coor_f(_ndimension);
    int        sc{};

    size_type block_v = 1;
    for(int d = 0; d < _ndimension; ++d) {
      block_r[d] = fine_->_rdimensions[d] / coarse_->_rdimensions[d];
      assert(block_r[d] * coarse_->_rdimensions[d] == fine_->_rdimensions[d]);
      block_v *= block_r[d];
    }

    lutVec_.resize(coarse_->oSites());
    lutPtr_.resize(coarse_->oSites());
    sizes_.resize(coarse_->oSites());
    reverseLutVec_.resize(fine_->oSites());
    for(index_type sc = 0; sc < lutVec_.size(); ++sc) {
      lutVec_[sc].resize(0);
      lutVec_[sc].reserve(block_v);
      lutPtr_[sc] = nullptr;
      sizes_[sc]  = 0;
    }

    // TODO: experiment if making this parallel with an omp critical is faster than this version
    for(index_type sf = 0; sf < fine_->oSites(); ++sf) {
      Lexicographic::CoorFromIndex(coor_f, sf, fine_->_rdimensions);
      for(int d = 0; d < _ndimension; ++d) coor_c[d] = coor_f[d] / block_r[d];
      Lexicographic::IndexFromCoor(coor_c, sc, coarse_->_rdimensions);
      lutVec_[sc].push_back(sf);
      sizes_[sc]++;
      reverseLutVec_[sf] = sc;
    }

    for(index_type sc = 0; sc < lutVec_.size(); ++sc) lutPtr_[sc] = &lutVec_[sc][0];

    isPopulated_ = true;
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
