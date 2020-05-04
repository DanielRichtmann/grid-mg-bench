/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_mrhs_clover.cc

    Copyright (C) 2015 - 2018

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
#include <Benchmark_helpers.h>

using namespace Grid;
using namespace Grid::BenchmarkHelpers;

struct WorkPerSite {
  double flop;
  double elem;
  double word = sizeof(Complex); // good default with possibility to override

  double byte() const { return elem * word; }
  double intensity() const { return elem == 0. ? 0. : flop/byte(); }
};

inline double cMatMulFlop(int n) { return ((8 - 2 / (double)n) * n * n); }

#define doComparison(name4d, name5d) \
  do { \
    LatticeFermion vec_tmp(UGrid); \
    LatticeFermion diff(UGrid); \
    \
    for(int i = 0; i < nvec; i++) { \
      ExtractSlice(vec_tmp, vecs_res_5d, i, 0); \
      diff = vecs_res_4d[i] - vec_tmp; \
      std::cout << GridLogMessage << "vector " << i << ": norm2(" << #name4d << "* v) = " << norm2(vecs_res_4d[i]) \
                << " norm2(" << #name5d << " * v) = " << norm2(vec_tmp) << " diff = " << norm2(diff) << std::endl; \
    } \
  } while(0)

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  int  nvec           = readFromCommandLineInt(&argc, &argv, "--nvec", 20);
  int  niter          = readFromCommandLineInt(&argc, &argv, "--niter", 1000);
  bool countPerformed = (GridCmdOptionExists(argv, argv + argc, "--performed")); // calculate performance with actually performed traffic rather than minimum required
  // clang-format on

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  // clang-format off
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian*         FGrid   = SpaceTimeGrid::makeFiveDimGrid(nvec, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(nvec, UGrid);
  // clang-format on

  std::cout << GridLogMessage << "UGrid:" << std::endl; UGrid->show_decomposition();
  std::cout << GridLogMessage << "UrbGrid:" << std::endl; UrbGrid->show_decomposition();
  std::cout << GridLogMessage << "FGrid:" << std::endl; FGrid->show_decomposition();
  std::cout << GridLogMessage << "FrbGrid:" << std::endl; FrbGrid->show_decomposition();

  GridParallelRNG  UPRNG(UGrid);
  GridParallelRNG  FPRNG(FGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  UPRNG.SeedFixedIntegers(seeds);
  FPRNG.SeedFixedIntegers(seeds);

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(UPRNG, Umu);

  RealD mass = 0.5;
  RealD csw  = 1.0;

  typename WilsonCloverFermionR::ImplParams implParams;
  WilsonAnisotropyCoefficients              anisParams;

  std::vector<Complex> boundary_phases(Nd, 1.);
  if(GridCmdOptionExists(argv, argv + argc, "--antiperiodic")) boundary_phases[Nd - 1] = -1.;
  implParams.boundary_phases = boundary_phases;

  WilsonFermionR Dw(Umu, *UGrid, *UrbGrid, mass, implParams, anisParams);
  WilsonCloverFermionR Dwc(Umu, *UGrid, *UrbGrid, mass, csw, csw, anisParams, implParams);

  WilsonMRHSFermionR Dw5(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, implParams);
  WilsonCloverMRHSFermionR Dwc5(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, implParams);

  /////////////////////////////////////////////////////////////////////////////
  //             Calculate numbers needed for performance figures            //
  /////////////////////////////////////////////////////////////////////////////

  double volumeUGrid = std::accumulate(UGrid->_fdimensions.begin(), UGrid->_fdimensions.end(), 1, std::multiplies<double>());
  double volumeFGrid = std::accumulate(FGrid->_fdimensions.begin(), FGrid->_fdimensions.end(), 1, std::multiplies<double>());

  WorkPerSite wilson_dhop, clover_dhop, twisted_mass_dhop;
  WorkPerSite wilson_diag, clover_diag, twisted_mass_diag;
  WorkPerSite wilson_diag_necessary, clover_diag_necessary, twisted_mass_diag_necessary;
  WorkPerSite wilson_diag_performed, clover_diag_performed, twisted_mass_diag_performed;
  WorkPerSite wilson_dir, clover_dir, twisted_mass_dir;
  WorkPerSite wilson_full, clover_full, twisted_mass_full;
  WorkPerSite spinor;
  WorkPerSite gaugemat;
  WorkPerSite clovmat;
  WorkPerSite clovmat_packed;

  spinor.elem         = Ns * Nc;
  gaugemat.elem       = Nc * Nc;
  clovmat.elem        = spinor.elem * spinor.elem; // full spincolor matrix
  clovmat_packed.elem = Nhs * (Nc + (Nhs*Nc) * (Nhs*Nc - 1) / 2); // 2 * (6 diag reals (= 3 complex) + 15 complex upper triang)

  wilson_dhop.flop = 8 * spinor.elem + 8*Nhs*cMatMulFlop(Nc) + 7 * (2*spinor.elem); // spin-proj + su3 x 2 half spinors + accum output
  wilson_dhop.elem = 8 * (gaugemat.elem + spinor.elem) + spinor.elem;

  clover_dhop.flop = wilson_dhop.flop;
  clover_dhop.elem = wilson_dhop.elem;

  twisted_mass_dhop.flop = wilson_dhop.flop;
  twisted_mass_dhop.elem = wilson_dhop.elem;

  wilson_diag_necessary.flop = 6 * spinor.elem;
  wilson_diag_performed.flop = 6 * spinor.elem;
  wilson_diag_necessary.elem = 2 * spinor.elem;
  wilson_diag_performed.elem = 2 * spinor.elem;

  clover_diag_necessary.flop = 2 * (cMatMulFlop(Nhs*Nc) - 4*Nhs*Nc); // 2 6x6 matmuls - diff between compl mul and real mul on diagonal
  clover_diag_performed.flop = cMatMulFlop(spinor.elem) * spinor.elem * spinor.elem; // clovmat x spinor

  clover_diag_necessary.elem = clovmat_packed.elem + 2 * spinor.elem;
  clover_diag_performed.elem = clovmat.elem + 2 * spinor.elem;

  twisted_mass_diag_necessary.flop = 3 * 2 * spinor.elem; // 2 real muls + a complex add
  twisted_mass_diag_performed.flop = 3 * 2 * spinor.elem; // 2 real muls + a complex add
  twisted_mass_diag_necessary.elem = 2 * spinor.elem;
  twisted_mass_diag_performed.elem = 2 * spinor.elem;

  if(countPerformed) { // count traffic that is actually performed by the hardware
    std::cout << GridLogMessage << "Using traffic value that is actually transfered for the full matrices" << std::endl;
    wilson_diag       = wilson_diag_performed;
    clover_diag       = clover_diag_performed;
    twisted_mass_diag = twisted_mass_diag_performed;
  } else { // count minimal traffic that is actually necessary (e.g., don't count the extra traffic for the spinors since it wouldn't be needed in a fused impl for M)
    std::cout << GridLogMessage << "Using minimal traffic value that is actually necessary for the full matrices" << std::endl;
    wilson_diag       = wilson_diag_necessary;
    clover_diag       = clover_diag_necessary;
    twisted_mass_diag = twisted_mass_diag_necessary;
  }

  wilson_dir.flop = spinor.elem + Nhs*cMatMulFlop(Nc); // spin project + su3 x half spinor + TODO reconstruct?
  wilson_dir.elem = gaugemat.elem + 2 * spinor.elem; // gauge mat + neigh spinor + output spinor

  clover_dir.flop = wilson_dir.flop;
  clover_dir.elem = wilson_dir.elem;

  twisted_mass_dir.flop = wilson_dir.flop;
  twisted_mass_dir.elem = wilson_dir.elem;

  wilson_full.flop = wilson_dhop.flop + wilson_diag.flop;
  wilson_full.elem = wilson_dhop.elem + wilson_diag.elem - 2*spinor.elem; // only requires one spinor load + store, not 2

  clover_full.flop = clover_dhop.flop + clover_diag.flop;
  clover_full.elem = clover_dhop.elem + clover_diag.elem - 2*spinor.elem; // only requires one spinor load + store, not 2

  twisted_mass_full.flop = twisted_mass_dhop.flop + twisted_mass_diag.flop;
  twisted_mass_full.elem = twisted_mass_dhop.elem + twisted_mass_diag.elem - 2*spinor.elem; // only requires one load + store, not 2

  // NOTE flop values per site from output of quda's dslash_test binary:
  // wilson = 1320
  // mobius = 1320
  // clover = 1824
  // twisted-mass = 1392
  // twisted-clover = 1872
  // domain-wall-4d = 1320
  // domain-wall = 1419
  // clover-hasenbusch-twist = 1824

  std::cout << GridLogPerformance << "4d volume: " << volumeUGrid << std::endl;
  std::cout << GridLogPerformance << "5d volume: " << volumeFGrid << std::endl;
  std::cout << GridLogPerformance << "Dw.Dhop flop/site, byte/site, flop/byte: " << wilson_dhop.flop << " " << wilson_dhop.byte() << " " << wilson_dhop.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwc.Dhop flop/site, byte/site, flop/byte: " << clover_dhop.flop << " " << clover_dhop.byte() << " " << clover_dhop.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwtm.Dhop flop/site, byte/site, flop/byte: " << twisted_mass_dhop.flop << " " << twisted_mass_dhop.byte() << " " << twisted_mass_dhop.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dw.Mdiag flop/site, byte/site, flop/byte: " << wilson_diag.flop << " " << wilson_diag.byte() << " " << wilson_diag.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwc.Mdiag flop/site, byte/site, flop/byte: " << clover_diag.flop << " " << clover_diag.byte() << " " << clover_diag.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwtm.Mdiag flop/site, byte/site, flop/byte: " << twisted_mass_diag.flop << " " << twisted_mass_diag.byte() << " " << twisted_mass_diag.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dw.Mdir flop/site, byte/site, flop/byte: " << wilson_dir.flop << " " << wilson_dir.byte() << " " << wilson_dir.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwc.Mdir flop/site, byte/site, flop/byte: " << clover_dir.flop << " " << clover_dir.byte() << " " << clover_dir.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwtm.Mdir flop/site, byte/site, flop/byte: " << twisted_mass_dir.flop << " " << twisted_mass_dir.byte() << " " << twisted_mass_dir.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dw.M flop/site, byte/site, flop/byte: " << wilson_full.flop << " " << wilson_full.byte() << " " << wilson_full.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwc.M flop/site, byte/site, flop/byte: " << clover_full.flop << " " << clover_full.byte() << " " << clover_full.intensity() << std::endl;
  std::cout << GridLogPerformance << "Dwtm.M flop/site, byte/site, flop/byte: " << twisted_mass_full.flop << " " << twisted_mass_full.byte() << " " << twisted_mass_full.intensity() << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  //                             Setup of vectors                            //
  /////////////////////////////////////////////////////////////////////////////

  std::vector<LatticeFermion> vecs_src_4d(nvec, UGrid);
  std::vector<LatticeFermion> vecs_res_4d(nvec, UGrid);

  LatticeFermion vecs_src_5d(FGrid);
  LatticeFermion vecs_res_5d(FGrid);

  for(int i=0; i<nvec; i++) {
    random(UPRNG, vecs_src_4d[i]);
    InsertSlice(vecs_src_4d[i], vecs_src_5d, i, 0);
  }

  /////////////////////////////////////////////////////////////////////////////
  //                           Start of benchmarks                           //
  /////////////////////////////////////////////////////////////////////////////

  // Wilson hopping term = "dslash" ///////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dw.Dhop(vecs_src_4d[i], vecs_res_4d[i], 0);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * wilson_dhop.flop;
    double byte = volumeUGrid * niter * nvec * wilson_dhop.byte();
    double intensity = wilson_dhop.intensity();

    std::cout << GridLogPerformance << "Performance Dw.Dhop: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Wilson hopping term = "dslash" ////////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      __SSC_START;
      Dw5.Dhop(vecs_src_5d, vecs_res_5d, 0);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeFGrid * niter * wilson_dhop.flop;
    double byte = volumeFGrid * niter * wilson_dhop.byte();
    double intensity = wilson_dhop.intensity();

    std::cout << GridLogPerformance << "Performance Dw5.Dhop: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of wilson Dhop /////////////////////////////////

  doComparison(Dw.Dhop, Dw5.Dhop);

  // Wilson diagonal term = "Mooee" ///////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dw.Mooee(vecs_src_4d[i], vecs_res_4d[i]);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * wilson_diag.flop;
    double byte = volumeUGrid * niter * nvec * wilson_diag.byte();
    double intensity = wilson_diag.intensity();

    std::cout << GridLogPerformance << "Performance Dw.Mooee: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Wilson diagonal term = "Mooee" ////////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      __SSC_START;
      Dw5.Mooee(vecs_src_5d, vecs_res_5d);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeFGrid * niter * wilson_diag.flop;
    double byte = volumeFGrid * niter * wilson_diag.byte();
    double intensity = wilson_diag.intensity();

    std::cout << GridLogPerformance << "Performance Dw5.Mooee: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of Wilson Mooee ////////////////////////////////

  doComparison(Dw.Mooee, Dw5.Mooee);

  // Wilson directional term = "Mdir" /////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dw.Mdir(vecs_src_4d[i], vecs_res_4d[i], 1, 0);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * wilson_dir.flop;
    double byte = volumeUGrid * niter * nvec * wilson_dir.byte();
    double intensity = wilson_dir.intensity();

    std::cout << GridLogPerformance << "Performance Dw.Mdir: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Wilson directional term = "Mdir" //////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      __SSC_START;
      Dw5.Mdir(vecs_src_5d, vecs_res_5d, 1, 0);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeFGrid * niter * wilson_dir.flop;
    double byte = volumeFGrid * niter * wilson_dir.byte();
    double intensity = wilson_dir.intensity();

    std::cout << GridLogPerformance << "Performance Dw5.Mdir: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of Wilson Mdir /////////////////////////////////

  doComparison(Dw.Mdir, Dw5.Mdir);

  // Wilson full matrix ///////////////////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dw.M(vecs_src_4d[i], vecs_res_4d[i]);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * wilson_full.flop;
    double byte = volumeUGrid * niter * nvec * wilson_full.byte();
    double intensity = wilson_full.intensity();

    std::cout << GridLogPerformance << "Performane Dw.M: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Wilson full matrix ////////////////////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n = 0; n < niter; n++) {
      __SSC_START;
      Dw5.M(vecs_src_5d, vecs_res_5d);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1 - t0) / 1e6;

    double flop = volumeFGrid * niter * wilson_full.flop;
    double byte = volumeFGrid * niter * wilson_full.byte();
    double intensity = wilson_full.intensity();

    std::cout << GridLogPerformance << "Performance Dw5.M: " << td << " s " << niter << " x " << intensity << " F/B " << flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of Wilson M ////////////////////////////////////

  doComparison(Dw.M, Dw5.M);

  // Clover diagonal term = "Mooee" ///////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dwc.Mooee(vecs_src_4d[i], vecs_res_4d[i]);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * clover_diag.flop;
    double byte = volumeUGrid * niter * nvec * clover_diag.byte();
    double intensity = clover_diag.intensity();

    std::cout << GridLogPerformance << "Performance Dwc.Mooee: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Clover diagonal term = "Mooee" ////////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      __SSC_START;
      Dwc5.Mooee(vecs_src_5d, vecs_res_5d);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeFGrid * niter * clover_diag.flop;
    double byte = volumeFGrid * niter * clover_diag.byte();
    double intensity = clover_diag.intensity();

    std::cout << GridLogPerformance << "Performance Dwc5.Mooee: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of Clover Mooee ////////////////////////////////

  doComparison(Dwc.Mooee, Dwc5.Mooee);

  // Clover full matrix ///////////////////////////////////////////////////////

  {
    for(int i=0; i<nvec; i++) {
      vecs_res_4d[i] = Zero();
    }

    double t0 = usecond();
    for(int n=0; n<niter; n++) {
      for(int i=0; i<nvec; i++) {
        __SSC_START;
        Dwc.M(vecs_src_4d[i], vecs_res_4d[i]);
        __SSC_STOP;
      }
    }
    double t1 = usecond();
    double td = (t1-t0)/1e6;

    double flop = volumeUGrid * niter * nvec * clover_full.flop;
    double byte = volumeUGrid * niter * nvec * clover_full.byte();
    double intensity = clover_full.intensity();

    std::cout << GridLogPerformance << "Performance Dwc.M: " << td << " s " << niter << " x " << intensity << " F/B "<< flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // 5d Clover full matrix ////////////////////////////////////////////////////

  {
    vecs_res_5d = Zero();

    double t0 = usecond();
    for(int n = 0; n < niter; n++) {
      __SSC_START;
      Dwc5.M(vecs_src_5d, vecs_res_5d);
      __SSC_STOP;
    }
    double t1 = usecond();
    double td = (t1 - t0) / 1e6;

    double flop = volumeFGrid * niter * clover_full.flop;
    double byte = volumeFGrid * niter * clover_full.byte();
    double intensity = clover_full.intensity();

    std::cout << GridLogPerformance << "Performance Dwc5.M: " << td << " s " << niter << " x " << intensity << " F/B " << flop/td << " F/s " << byte/td << " B/s" << std::endl;
  }

  // Compare 4d and 5d version of clover M ////////////////////////////////////

  doComparison(Dwc.M, Dwc5.M);

  Grid_finalize();
}
