/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/multigrid/Multigrid.h

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
/*  ENDLEGAL */

#pragma once

#include <tests/multigrid/CoarseningPolicyRework.h>
#include <tests/multigrid/CoarseningLookupTableRework.h>
#include <tests/multigrid/AggregationKernelsRework.h>
#include <tests/multigrid/CoarsenedMatrixKernelsRework.h>
#include <tests/multigrid/Profiler.h>
#include <tests/multigrid/BasisVectors.h>
#include <tests/multigrid/AggregationRework.h>
#include <tests/multigrid/CoarsenedMatrixRework.h>
#include <tests/multigrid/MultigridParams.h>
#include <tests/multigrid/Helpers.h>
#include <tests/multigrid/LevelInfo.h>
#include <tests/multigrid/MultigridPreconditioner.h>
