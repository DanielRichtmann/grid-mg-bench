/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/MultiGridPreconditioner.h

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

#define vectorViewPointerOpen(v, p, l, mode)     \
  Vector<decltype(l[0].View(mode))> v; \
  v.reserve(l.size()); \
  for(uint64_t k=0; k<l.size(); k++) v.push_back(l[k].View(mode)); \
  typename std::remove_reference<decltype(v[0])>::type* p = &v[0];

#define vectorViewPointerClose(v, p) \
  for(uint64_t k=0; k<v.size(); k++) v[k].ViewClose(); \
  p = nullptr;

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
