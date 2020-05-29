/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_helpers.h

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
#ifndef GRID_BENCHMARK_HELPERS_H
#define GRID_BENCHMARK_HELPERS_H

#include "Grid/util/Profiling.h"

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(BenchmarkHelpers);

#define BenchmarkFunction(function, flop, byte, nIterMin, nSecMin, ...) \
  do { \
    uint64_t nIterFinal = 1; \
    if(nIterMin != 1) { \
      uint64_t nIterInitial = 10; \
      double t0 = usecond(); \
      for(uint64_t i = 0; i < nIterInitial; ++i) { \
        __SSC_START; \
        function(__VA_ARGS__); \
        __SSC_STOP; \
      } \
      double td  = (usecond()-t0)/1e6; \
      nIterFinal = std::min(nIterMin, (uint64_t)(nSecMin*nIterInitial/td)+1); \
    } \
    Profiler prof; \
    prof.Start(#function, flop, byte); \
    for(uint64_t i = 0; i < nIterFinal; ++i) { \
      __SSC_START; \
      function(__VA_ARGS__); \
      __SSC_STOP; \
    } \
    prof.Stop(#function, nIterFinal); \
    prettyPrintProfiling("Kernel", prof.GetResults(), GridTime(0), true); \
  } while(0)

#define BenchmarkFunctionMRHS(function, flop, byte, nIterMin, nSecMin, nRHS, ...) \
  do { \
    uint64_t nIterFinal = 1; \
    if(nIterMin != 1) { \
      uint64_t nIterInitial = 10; \
      double t0 = usecond(); \
      for(uint64_t i = 0; i < nIterInitial; ++i) { \
        for(uint64_t rhs = 0; rhs < nRHS; ++rhs) { \
          __SSC_START; \
          function(__VA_ARGS__); \
          __SSC_STOP; \
        } \
      } \
      double td  = (usecond()-t0)/1e6; \
      nIterFinal = std::min(nIterMin, (uint64_t)(nSecMin*nIterInitial/td)+1); \
    } \
    Profiler prof; \
    prof.Start(#function, flop, byte); \
    for(uint64_t i = 0; i < nIterFinal; ++i) { \
      for(uint64_t rhs = 0; rhs < nRHS; ++rhs) { \
        __SSC_START; \
        function(__VA_ARGS__); \
        __SSC_STOP; \
      } \
    } \
    prof.Stop(#function, nIterFinal); \
    prettyPrintProfiling("Kernel", prof.GetResults(), GridTime(0), true); \
  } while(0)

int readFromCommandLineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}

Coordinate readFromCommandLineCoordinate(int*               argc,
                                         char***            argv,
                                         const std::string& option,
                                         const Coordinate&  defaultValues) {
  std::string      arg;
  std::vector<int> ret(defaultValues.toVector());
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionIntVector(arg, ret);
  }
  return ret;
}

bool readFromCommandLineToggle(int* argc, char*** argv, const std::string& option) {
  return GridCmdOptionExists(*argv, *argv + *argc, option);
}

std::vector<std::string> readFromCommandLineCSL(int*                            argc,
                                                char***                         argv,
                                                const std::string&              option,
                                                const std::vector<std::string>& defaultValues) {
  std::string      arg;
  std::vector<std::string> ret(defaultValues);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionCSL(arg, ret);
  }
  return ret;
}

template<typename Field>
void performChiralDoublingG5R5(std::vector<Field>& basisVectors) {
  assert(basisVectors.size()%2 == 0);
  auto nb = basisVectors.size()/2;
  for(int n=0; n<nb; n++) {
    auto tmp1 = basisVectors[n];
    auto tmp2 = tmp1;
    G5R5(tmp2, basisVectors[n]);
    axpby(basisVectors[n], 1.0, 1.0, tmp1, tmp2);
    axpby(basisVectors[n+nb], 1.0, -1.0, tmp1, tmp2);
    std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
              << "norm2(vec[" << n << "]) = " << norm2(basisVectors[n]) << ". "
              << "norm2(vec[" << n+nb << "]) = " << norm2(basisVectors[n+nb]) << std::endl;

    auto tmp1_v    = tmp1.View();
    auto basis_l_v = basisVectors[n].View();
    auto basis_r_v = basisVectors[n+nb].View();

    std::cout << GridLogDebug << "original = " << tmp1_v[0] << std::endl;
    std::cout << GridLogDebug << "left = " << basis_l_v[0] << std::endl;
    std::cout << GridLogDebug << "right = " << basis_r_v[0] << std::endl;
  }
}

template<typename Field>
void performChiralDoublingG5C(std::vector<Field>& basisVectors) {
  assert(basisVectors.size()%2 == 0);
  auto nb = basisVectors.size()/2;

  for(int n=0; n<nb; n++) {
    auto tmp1 = basisVectors[n];
    auto tmp2 = tmp1;
    G5C(tmp2, basisVectors[n]);
    axpby(basisVectors[n], 0.5, 0.5, tmp1, tmp2);
    axpby(basisVectors[n+nb], 0.5, -0.5, tmp1, tmp2);
    std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
              << "norm2(vec[" << n << "]) = " << norm2(basisVectors[n]) << ". "
              << "norm2(vec[" << n+nb << "]) = " << norm2(basisVectors[n+nb]) << std::endl;

    auto tmp1_v    = tmp1.View();
    auto basis_l_v = basisVectors[n].View();
    auto basis_r_v = basisVectors[n+nb].View();

    std::cout << GridLogDebug << "original = " << tmp1_v[0] << std::endl;
    std::cout << GridLogDebug << "left = " << basis_l_v[0] << std::endl;
    std::cout << GridLogDebug << "right = " << basis_r_v[0] << std::endl;
  }
}

// template<typename Field>
// void performChiralDoubling(std::vector<Field>& basisVectors) {
//   GridBase* grid = basisVectors[0].Grid();
//   auto ndim = grid->_ndimension;
//   if(ndim == 5 && grid->_rdimensions[0] != 1)
//     performChiralDoublingG5R5(basisVectors);
//   else
//     performChiralDoublingG5C(basisVectors);
// }

template<typename Field>
void undoChiralDoublingG5R5(std::vector<Field>& basisVectors) {
  assert(basisVectors.size()%2 == 0);
  auto nb = basisVectors.size()/2;

  for(int n=0; n<nb; n++) {
    std::cout << GridLogWarning << "undoChiralDoubling5d not implemented yet" << std::endl;
  }
}

template<typename Field>
void undoChiralDoublingG5C(std::vector<Field>& basisVectors) {
  assert(basisVectors.size()%2 == 0);
  auto nb = basisVectors.size()/2;

  for(int n=0; n<nb; n++) {
    basisVectors[n] = basisVectors[n] + basisVectors[n+nb];
    std::cout << GridLogMessage << "Undid chiral doubling of vector " << n << ". "
              << "norm2(vec[" << n << "]) = " << norm2(basisVectors[n]) << std::endl;
  }
}

template<class Field>
void assertResultMatchesReference(RealD tolerance, Field const& reference, Field const& result) {
  conformable(reference.Grid(), result.Grid());
  Field diff(reference.Grid());

  diff        = reference - result;
  auto absDev = norm2(diff);
  auto relDev = absDev / norm2(reference);

  std::cout << GridLogMessage
            << "ref = " << norm2(reference)
            << " res = " << norm2(result)
            << " absolute deviation = " << absDev
            << " relative deviation = " << relDev;

  if(relDev <= tolerance) {
    std::cout << " < " << tolerance << " -> check passed" << std::endl;
  } else {
    std::cout << " > " << tolerance << " -> check failed" << std::endl;
  }
  assert(relDev <= tolerance);
}

template<class vobj>
int getSiteElems() {
  int ret = 1;

  typedef typename getVectorType<vobj>::type vobj_site; // gives us the the type of the site object if vobj is a Lattice type

  // TODO: This doesn't work for arbitry tensors. One would need to get the tensor recursion depth and then walk over these and multiply the values

  int _ColourN       = indexRank<ColourIndex,vobj_site>();
  int _ColourScalar  =  isScalar<ColourIndex,vobj_site>();
  int _ColourVector  =  isVector<ColourIndex,vobj_site>();
  int _ColourMatrix  =  isMatrix<ColourIndex,vobj_site>();

  int _SpinN         = indexRank<SpinIndex,vobj_site>();
  int _SpinScalar    =  isScalar<SpinIndex,vobj_site>();
  int _SpinVector    =  isVector<SpinIndex,vobj_site>();
  int _SpinMatrix    =  isMatrix<SpinIndex,vobj_site>();

  int _LorentzN      = indexRank<LorentzIndex,vobj_site>();
  int _LorentzScalar =  isScalar<LorentzIndex,vobj_site>();
  int _LorentzVector =  isVector<LorentzIndex,vobj_site>();
  int _LorentzMatrix =  isMatrix<LorentzIndex,vobj_site>();

  if (_ColourMatrix) ret *= _ColourN * _ColourN;
  else               ret *= _ColourN;

  if (_SpinMatrix) ret *= _SpinN * _SpinN;
  else             ret *= _SpinN;

  if (_LorentzMatrix) ret *= _LorentzN * _LorentzN;
  else                ret *= _LorentzN;

  return ret;
}

NAMESPACE_END(BenchmarkHelpers);
NAMESPACE_END(Grid);

#endif
