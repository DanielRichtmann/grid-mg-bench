    // blockZaxpy in bockPromote - 3s, 5%
    // noncoalesced linalg in Preconditionoer ~ 3s 5%
    // Lancos tuning or replace 10-20s ~ 25%, open ended
    // setup tuning   5s  ~  8%
    //    -- e.g. ordermin, orderstep tunables.
    // MdagM path without norm in LinOp code.     few seconds

    // Mdir calc blocking kernels
    // Fuse kernels in blockMaskedInnerProduct
    // preallocate Vectors in Cayley 5D ~ few percent few seconds

/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/algorithms/CoarsenedMatrix.h

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>

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
#ifndef  GRID_ALGORITHM_COARSENED_MATRIX_UPSTREAM_IMPROVED_DIRSAVE_LUT_H
#define  GRID_ALGORITHM_COARSENED_MATRIX_UPSTREAM_IMPROVED_DIRSAVE_LUT_H


NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(UpstreamImprovedDirsaveLut);

template<class vobj,class CComplex>
inline void blockMaskedInnerProduct(Lattice<CComplex> &CoarseInner,
				    const Lattice<decltype(innerProduct(vobj(),vobj()))> &FineMask,
				    const Lattice<vobj> &fineX,
				    const Lattice<vobj> &fineY)
{
  typedef decltype(innerProduct(vobj(),vobj())) dotp;

  GridBase *coarse(CoarseInner.Grid());
  GridBase *fine  (fineX.Grid());

  Lattice<dotp> fine_inner(fine); fine_inner.Checkerboard() = fineX.Checkerboard();
  Lattice<dotp> fine_inner_msk(fine);

  // Multiply could be fused with innerProduct
  // Single block sum kernel could do both masks.
  fine_inner = localInnerProduct(fineX,fineY);
  mult(fine_inner_msk, fine_inner,FineMask);
  blockSum(CoarseInner,fine_inner_msk);
}

template<class vobj, class CComplex, int nbasis>
inline void blockLutedInnerProduct(Lattice<iVector<CComplex, nbasis>>  &coarseData,
                                   const Lattice<vobj>                 &fineData,
                                   const std::vector<Lattice<vobj>>    &Basis,
                                   const Rework::CoarseningLookupTable &lut)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  int ndimU = Basis[0].Grid()->_ndimension;
  int ndimF = coarse->_ndimension;
  int LLs   = 1;

  // checks
  assert(fine->_ndimension == ndimF);
  assert(ndimF == ndimU || ndimF == ndimU+1);
  assert(nbasis == Basis.size());
  if(ndimF == ndimU) { // strictly 4d or strictly 5d
    assert(lut.gridPointersMatch(coarse, fine));
    for(auto const& elem : Basis) conformable(elem, fineData);
    LLs = 1;
  } else if(ndimF == ndimU+1) { // 4d with mrhs via 5th dimension
    assert(coarse->_rdimensions[0] == fine->_rdimensions[0]);   // same extent in 5th dimension
    assert(coarse->_fdimensions[0] == coarse->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
    assert(fine->_fdimensions[0]   == fine->_rdimensions[0]);   // 5th dimension strictly local and not cb'ed
    LLs = coarse->_rdimensions[0];
  }

  auto lut_v        = lut.View();
  auto sizes_v      = lut.Sizes();
  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  vectorViewPointerOpen(Basis_v, Basis_p, Basis, AcceleratorRead);

  accelerator_for(scFi, nbasis*coarse->oSites(), vobj::Nsimd(), {
    auto i   = scFi%nbasis;
    auto scF = scFi/nbasis;
    auto s5  = scF%LLs;
    auto scU = scF/LLs;

    decltype(innerProduct(Basis_p[0](0), fineData_v(0))) reduce = Zero();

    for(int j=0; j<sizes_v[scU]; ++j) {
      int sfU = lut_v[scU][j];
      int sfF = sfU*LLs + s5;
      reduce = reduce + innerProduct(Basis_p[i](sfU), fineData_v(sfF));
    }
    coalescedWrite(coarseData_v[scF](i), reduce);
  });

  vectorViewPointerClose(Basis_v, Basis_p);
}

template<class vobj, class CComplex, int nbasis>
inline void blockLutedAxpy(const Lattice<iVector<CComplex, nbasis>> &coarseData,
                           Lattice<vobj>                            &fineData,
                           const std::vector<Lattice<vobj>>         &Basis,
                           const Rework::CoarseningLookupTable      &lut) {
  GridBase *fine = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  int ndimU = Basis[0].Grid()->_ndimension;
  int ndimF = coarse->_ndimension;
  int LLs   = 1;

  // checks
  assert(fine->_ndimension == ndimF);
  assert(ndimF == ndimU || ndimF == ndimU+1);
  assert(nbasis == Basis.size());
  if(ndimF == ndimU) { // strictly 4d or strictly 5d
    assert(lut.gridPointersMatch(coarse, fine));
    for(auto const& elem : Basis) conformable(elem, fineData);
    LLs = 1;
  } else if(ndimF == ndimU+1) { // 4d with mrhs via 5th dimension
    assert(coarse->_rdimensions[0] == fine->_rdimensions[0]);   // same extent in 5th dimension
    assert(coarse->_fdimensions[0] == coarse->_rdimensions[0]); // 5th dimension strictly local and not cb'ed
    assert(fine->_fdimensions[0]   == fine->_rdimensions[0]);   // 5th dimension strictly local and not cb'ed
    LLs = coarse->_rdimensions[0];
  }

  auto rlut_v       = lut.ReverseView();
  autoView(fineData_v, fineData, AcceleratorWrite);
  autoView(coarseData_v, coarseData, AcceleratorRead);

  vectorViewPointerOpen(Basis_v, Basis_p, Basis, AcceleratorRead);

  accelerator_for(sfF, fine->oSites(), vobj::Nsimd(), {
    auto s5  = sfF%LLs;
    auto sfU = sfF/LLs;
    auto scU = rlut_v[sfU];
    auto scF = scU*LLs + s5;

    auto fineData_t = fineData_v(sfF);
    for(int i=0; i<nbasis; ++i) {
      if(i == 0)
        fineData_t = coarseData_v(scF)(i) * Basis_p[i](sfU);
      else
        fineData_t = fineData_t + coarseData_v(scF)(i) * Basis_p[i](sfU);
    }
    coalescedWrite(fineData_v[sfF], fineData_t);
  });

  vectorViewPointerClose(Basis_v, Basis_p);
}

template<class CComplex, class vobj>
void blockLutedOrthonormalise(Lattice<CComplex>                    &ip,
                              std::vector<Lattice<vobj>>           &Basis,
                              const Rework::CoarseningLookupTable  &lut) {
  GridBase *fine   = Basis[0].Grid();
  GridBase *coarse = ip.Grid();

  int nbasis = Basis.size();

  subdivides(coarse, fine);
  for(int i=0; i<nbasis; ++i) conformable(Basis[i].Grid(), fine);

  typedef decltype(innerProduct(vobj(), vobj()))   dotp;
  typedef Rework::CoarseningLookupTable::size_type size_type;

  Lattice<dotp> alpha(coarse);
  Lattice<dotp> norm(coarse);

  autoView( alpha_v, alpha, AcceleratorRead);
  autoView( norm_v, norm, AcceleratorRead);
  auto  lut_v    = lut.View();
  auto  sizes_v  = lut.Sizes();
  vectorViewPointerOpen(Basis_v, Basis_p, Basis, AcceleratorWrite);

  // Kernel fusion
  accelerator_for(sc, coarse->oSites(), vobj::Nsimd(), {
    auto alpha_t = alpha_v(sc);
    auto norm_t  = norm_v(sc);

    for(int v=0; v<nbasis; ++v) {
      for(int u=0; u<v; ++u) {
        alpha_t = Zero();

        // alpha = <basis[u], basis[v]>
        for(size_type i=0; i<sizes_v[sc]; ++i) {
          auto sf   = lut_v[sc][i];
          alpha_t = alpha_t + innerProduct(Basis_p[u](sf), Basis_p[v](sf));
        }

        // basis[v] -= alpha * basis[u]
        for(size_type i=0; i<sizes_v[sc]; ++i) {
          auto sf = lut_v[sc][i];
          coalescedWrite(Basis_p[v][sf], Basis_p[v](sf) - alpha_t * Basis_p[u](sf));
        }
      }

      norm_t = Zero();

      // norm = <basis[v], basis[v]>
      for(size_type i=0; i<sizes_v[sc]; ++i) {
        auto sf = lut_v[sc][i];
        norm_t  = norm_t + innerProduct(Basis_p[v](sf), Basis_p[v](sf));
      }

      norm_t = pow(norm_t, -0.5);

      // basis[v] = 1/norm * basis[v]
      for(size_type i=0; i<sizes_v[sc]; ++i) {
        auto sf = lut_v[sc][i];
        coalescedWrite(Basis_p[v][sf], norm_t * Basis_p[v](sf));
      }
    }
  });
  vectorViewPointerClose(Basis_v, Basis_p);
}

class Geometry {
public:
  int npoint;
  std::vector<int> directions   ;
  std::vector<int> displacements;

  Geometry(int _d)  {
    
    int base = (_d==5) ? 1:0;

    // make coarse grid stencil for 4d , not 5d
    if ( _d==5 ) _d=4;

    npoint = 2*_d+1;
    directions.resize(npoint);
    displacements.resize(npoint);
    for(int d=0;d<_d;d++){
      directions[d   ] = d+base;
      directions[d+_d] = d+base;
      displacements[d  ] = +1;
      displacements[d+_d]= -1;
    }
    directions   [2*_d]=0;
    displacements[2*_d]=0;
      
    //// report back
    std::cout<<GridLogMessage<<"directions    :";
    for(int d=0;d<npoint;d++) std::cout<< directions[d]<< " ";
    std::cout<<std::endl;
    std::cout<<GridLogMessage<<"displacements :";
    for(int d=0;d<npoint;d++) std::cout<< displacements[d]<< " ";
    std::cout<<std::endl;
  }
  
  /*
  // Original cleaner code
  Geometry(int _d) : dimension(_d), npoint(2*_d+1), directions(npoint), displacements(npoint) {
  for(int d=0;d<dimension;d++){
  directions[2*d  ] = d;
  directions[2*d+1] = d;
  displacements[2*d  ] = +1;
  displacements[2*d+1] = -1;
  }
  directions   [2*dimension]=0;
  displacements[2*dimension]=0;
  }
  std::vector<int> GetDelta(int point) {
  std::vector<int> delta(dimension,0);
  delta[directions[point]] = displacements[point];
  return delta;
  };
  */    

};
  
template<class Fobj,class CComplex,int nbasis>
class Aggregation   {
public:
  typedef iVector<CComplex,nbasis >             siteVector;
  typedef Lattice<siteVector>                 CoarseVector;
  typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;

  typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
  typedef Lattice<Fobj >        FineField;

  GridBase *CoarseGrid;
  GridBase *FineGrid;
  std::vector<Lattice<Fobj> > subspace;
  int checkerboard;
  Grid::Rework::CoarseningLookupTable lut;
  int Checkerboard(void){return checkerboard;}
  Aggregation(GridBase *_CoarseGrid,GridBase *_FineGrid,int _checkerboard) : 
    CoarseGrid(_CoarseGrid),
    FineGrid(_FineGrid),
    subspace(nbasis,_FineGrid),
    checkerboard(_checkerboard),
    lut(CoarseGrid, FineGrid)
  {
  };
  
  void Orthogonalise(int checkOrthogonality = 1, int passes = 2){
    CoarseScalar InnerProd(CoarseGrid); 
    for(int n = 0; n < passes; ++n) {
      std::cout << GridLogMessage <<" Block Gramm-Schmidt pass "<<n+1<<std::endl;
      blockLutedOrthonormalise(InnerProd,subspace,lut);
    }
    if(checkOrthogonality) CheckOrthogonal();
  }
  void CheckOrthogonal(void){
    CoarseVector iProj(CoarseGrid); 
    CoarseVector eProj(CoarseGrid); 
    for(int i=0;i<nbasis;i++){
      blockProject(iProj,subspace[i],subspace);
      eProj=Zero(); 
      autoView(eProj_v, eProj, AcceleratorWrite);
      accelerator_for(ss, CoarseGrid->oSites(),1,{
	eProj_v[ss](i)=CComplex(1.0);
      });
      eProj=eProj - iProj;
      std::cout<<GridLogMessage<<"Orthog check error "<<i<<" " << norm2(eProj)<<std::endl;
    }
    std::cout<<GridLogMessage <<"CheckOrthog done"<<std::endl;
  }
  void ProjectToSubspace(CoarseVector &CoarseVec,const FineField &FineVec){
    blockLutedInnerProduct(CoarseVec,FineVec,subspace,lut);
  }
  void PromoteFromSubspace(const CoarseVector &CoarseVec,FineField &FineVec){
    FineVec.Checkerboard() = subspace[0].Checkerboard();
    blockLutedAxpy(CoarseVec,FineVec,subspace,lut);
  }
  void CreateSubspaceRandom(GridParallelRNG &RNG){
    for(int i=0;i<nbasis;i++){
      random(RNG,subspace[i]);
    }
  }

  virtual void CreateSubspace(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis) {

    RealD scale;

    ConjugateGradient<FineField> CG(1.0e-2,100,false);
    FineField noise(FineGrid);
    FineField Mn(FineGrid);

    for(int b=0;b<nn;b++){
	
      subspace[b] = Zero();
      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5); 
      noise=noise*scale;
	
      hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise   ["<<b<<"] <n|MdagM|n> "<<norm2(Mn)<<std::endl;

      for(int i=0;i<1;i++){

	CG(hermop,noise,subspace[b]);

	noise = subspace[b];
	scale = std::pow(norm2(noise),-0.5); 
	noise=noise*scale;

      }

      hermop.Op(noise,Mn); std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(Mn)<<std::endl;
      subspace[b]   = noise;

    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // World of possibilities here. But have tried quite a lot of experiments (250+ jobs run on Summit)
  // and this is the best I found
  ////////////////////////////////////////////////////////////////////////////////////////////////
#if 1
  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo,
				       int orderfilter,
				       int ordermin,
				       int orderstep,
				       double filterlo
				       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);

    // New normalised noise
    gaussian(RNG,noise);
    scale = std::pow(norm2(noise),-0.5); 
    noise=noise*scale;

    // Initial matrix element
    hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

    int b =0;
    {
      // Filter
      Chebyshev<FineField> Cheb(lo,hi,orderfilter);
      Cheb(hermop,noise,Mn);
      // normalise
      scale = std::pow(norm2(Mn),-0.5); 	Mn=Mn*scale;
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp); 
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
      b++;
    }

    // Generate a full sequence of Chebyshevs
    {
      lo=filterlo;
      noise=Mn;

      FineField T0(FineGrid); T0 = noise;  
      FineField T1(FineGrid); 
      FineField T2(FineGrid);
      FineField y(FineGrid);
      
      FineField *Tnm = &T0;
      FineField *Tn  = &T1;
      FineField *Tnp = &T2;

      // Tn=T1 = (xscale M + mscale)in
      RealD xscale = 2.0/(hi-lo);
      RealD mscale = -(hi+lo)/(hi-lo);
      hermop.HermOp(T0,y);
      T1=y*xscale+noise*mscale;

      for(int n=2;n<=ordermin+orderstep*(nn-2);n++){
	
	hermop.HermOp(*Tn,y);

	autoView( y_v , y, AcceleratorWrite);
	autoView( Tn_v , (*Tn), AcceleratorWrite);
	autoView( Tnp_v , (*Tnp), AcceleratorWrite);
	autoView( Tnm_v , (*Tnm), AcceleratorWrite);
	const int Nsimd = CComplex::Nsimd();
	accelerator_forNB(ss, FineGrid->oSites(), Nsimd, {
	  coalescedWrite(y_v[ss],xscale*y_v(ss)+mscale*Tn_v(ss));
	  coalescedWrite(Tnp_v[ss],2.0*y_v(ss)-Tnm_v(ss));
        });

	// Possible more fine grained control is needed than a linear sweep,
	// but huge productivity gain if this is simple algorithm and not a tunable
	int m =1;
	if ( n>=ordermin ) m=n-ordermin;
	if ( (m%orderstep)==0 ) { 
	  Mn=*Tnp;
	  scale = std::pow(norm2(Mn),-0.5);         Mn=Mn*scale;
	  subspace[b] = Mn;
	  hermop.Op(Mn,tmp); 
	  std::cout<<GridLogMessage << n<<" filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl;
	  b++;
	}

	// Cycle pointers to avoid copies
	FineField *swizzle = Tnm;
	Tnm    =Tn;
	Tn     =Tnp;
	Tnp    =swizzle;
	  
      }
    }
    assert(b==nn);
  }
#endif
#if 0
  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo,
				       int orderfilter,
				       int ordermin,
				       int orderstep,
				       double filterlo
				       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);
    FineField combined(FineGrid);

    // New normalised noise
    gaussian(RNG,noise);
    scale = std::pow(norm2(noise),-0.5); 
    noise=noise*scale;

    // Initial matrix element
    hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

    int b =0;
#define FILTERb(llo,hhi,oorder)						\
    {									\
      Chebyshev<FineField> Cheb(llo,hhi,oorder);			\
      Cheb(hermop,noise,Mn);						\
      scale = std::pow(norm2(Mn),-0.5); Mn=Mn*scale;			\
      subspace[b]   = Mn;						\
      hermop.Op(Mn,tmp);						\
      std::cout<<GridLogMessage << oorder<< " Cheb filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl; \
      b++;								\
    }									

    //      JacobiPolynomial<FineField> Cheb(0.002,60.0,1500,-0.5,3.5);	\

    RealD alpha=-0.8;
    RealD beta =-0.8;
#define FILTER(llo,hhi,oorder)						\
    {									\
      Chebyshev<FineField> Cheb(llo,hhi,oorder);			\
      /* JacobiPolynomial<FineField> Cheb(0.0,60.0,oorder,alpha,beta);*/\
      Cheb(hermop,noise,Mn);						\
      scale = std::pow(norm2(Mn),-0.5); Mn=Mn*scale;			\
      subspace[b]   = Mn;						\
      hermop.Op(Mn,tmp);						\
      std::cout<<GridLogMessage << oorder<< "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl; \
      b++;								\
    }									
    
#define FILTERc(llo,hhi,oorder)				\
    {							\
      Chebyshev<FineField> Cheb(llo,hhi,oorder);	\
      Cheb(hermop,noise,combined);			\
    }									

    double node = 0.000;
    FILTERb(lo,hi,orderfilter);// 0
    //    FILTERc(node,hi,51);// 0
    noise = Mn;
    int base = 0;
    int mult = 100;
    FILTER(node,hi,base+1*mult);
    FILTER(node,hi,base+2*mult);
    FILTER(node,hi,base+3*mult);
    FILTER(node,hi,base+4*mult);
    FILTER(node,hi,base+5*mult);
    FILTER(node,hi,base+6*mult);
    FILTER(node,hi,base+7*mult);
    FILTER(node,hi,base+8*mult);
    FILTER(node,hi,base+9*mult);
    FILTER(node,hi,base+10*mult);
    FILTER(node,hi,base+11*mult);
    FILTER(node,hi,base+12*mult);
    FILTER(node,hi,base+13*mult);
    FILTER(node,hi,base+14*mult);
    FILTER(node,hi,base+15*mult);
    assert(b==nn);
  }
#endif

#if 0
  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,
				       int nn,
				       double hi,
				       double lo,
				       int orderfilter,
				       int ordermin,
				       int orderstep,
				       double filterlo
				       ) {

    RealD scale;

    FineField noise(FineGrid);
    FineField Mn(FineGrid);
    FineField tmp(FineGrid);
    FineField combined(FineGrid);

    // New normalised noise
    gaussian(RNG,noise);
    scale = std::pow(norm2(noise),-0.5); 
    noise=noise*scale;

    // Initial matrix element
    hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise <n|MdagM|n> "<<norm2(Mn)<<std::endl;

    int b =0;
    {						
      Chebyshev<FineField> JacobiPoly(0.005,60.,1500);
      //      JacobiPolynomial<FineField> JacobiPoly(0.002,60.0,1500,-0.5,3.5);
      //JacobiPolynomial<FineField> JacobiPoly(0.03,60.0,500,-0.5,3.5);
      //      JacobiPolynomial<FineField> JacobiPoly(0.00,60.0,1000,-0.5,3.5);
      JacobiPoly(hermop,noise,Mn);
      scale = std::pow(norm2(Mn),-0.5); Mn=Mn*scale;
      subspace[b]   = Mn;
      hermop.Op(Mn,tmp);
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl; 
      b++;
      //      scale = std::pow(norm2(tmp),-0.5);     tmp=tmp*scale;
      //      subspace[b]   = tmp;      b++;
      //    }									
    }									

#define FILTER(lambda)						\
    {								\
      hermop.HermOp(subspace[0],tmp);				\
      tmp = tmp - lambda *subspace[0];				\
      scale = std::pow(norm2(tmp),-0.5);			\
      tmp=tmp*scale;							\
      subspace[b]   = tmp;						\
      hermop.Op(subspace[b],tmp);					\
      std::cout<<GridLogMessage << "filt ["<<b<<"] <n|MdagM|n> "<<norm2(tmp)<<std::endl; \
      b++;								\
    }									
    //      scale = std::pow(norm2(tmp),-0.5);     tmp=tmp*scale;
    //      subspace[b]   = tmp;      b++;
    //    }									

    FILTER(2.0e-5);
    FILTER(2.0e-4);
    FILTER(4.0e-4);
    FILTER(8.0e-4);
    FILTER(8.0e-4);

    FILTER(2.0e-3);
    FILTER(3.0e-3);
    FILTER(4.0e-3);
    FILTER(5.0e-3);
    FILTER(6.0e-3);

    FILTER(2.5e-3);
    FILTER(3.5e-3);
    FILTER(4.5e-3);
    FILTER(5.5e-3);
    FILTER(6.5e-3);

    //    FILTER(6.0e-5);//6
    //    FILTER(7.0e-5);//8
    //    FILTER(8.0e-5);//9
    //    FILTER(9.0e-5);//3

    /*
    //    FILTER(1.0e-4);//10
    FILTER(2.0e-4);//11
    //   FILTER(3.0e-4);//12
    //    FILTER(4.0e-4);//13
    FILTER(5.0e-4);//14

    FILTER(6.0e-3);//4
    FILTER(7.0e-4);//1
    FILTER(8.0e-4);//7
    FILTER(9.0e-4);//15
    FILTER(1.0e-3);//2

    FILTER(2.0e-3);//2
    FILTER(3.0e-3);//2
    FILTER(4.0e-3);//2
    FILTER(5.0e-3);//2
    FILTER(6.0e-3);//2

    FILTER(7.0e-3);//2
    FILTER(8.0e-3);//2
    FILTER(1.0e-2);//2
    */
    std::cout << GridLogMessage <<"Jacobi filtering done" <<std::endl;
    assert(b==nn);
  }
#endif


};

// Fine Object == (per site) type of fine field
// nbasis      == number of deflation vectors
template<class Fobj,class CComplex,int nbasis>
class CoarsenedMatrix : public SparseMatrixBase<Lattice<iVector<CComplex,nbasis > > >
                      , public Rework::Profileable {
public:
    
  typedef iVector<CComplex,nbasis >           siteVector;
  typedef Lattice<CComplex >                  CoarseComplexField;
  typedef Lattice<siteVector>                 CoarseVector;
  typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;
  typedef iMatrix<CComplex,nbasis >  Cobj;
  typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
  typedef Lattice<Fobj >        FineField;

  ////////////////////
  // Data members
  ////////////////////
  Geometry         geom;
  GridBase *       _grid; 
  int hermitian;

  CartesianStencil<siteVector,siteVector,int> Stencil; 

  std::vector<CoarseMatrix> A;
      
  ///////////////////////
  // Interface
  ///////////////////////
  GridBase * Grid(void)         { return _grid; };   // this is all the linalg routines need to know

  void M (const CoarseVector &in, CoarseVector &out)
  {
    conformable(_grid,in.Grid());
    conformable(in.Grid(),out.Grid());

    SimpleCompressor<siteVector> compressor;

    double comms_usec = -usecond();
    Stencil.HaloExchange(in,compressor);
    comms_usec += usecond();

    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    autoView(Stencil_v, Stencil, AcceleratorRead);
    auto& geom_ = geom;
    vectorViewPointerOpen(Aview_v, Aview_p, A, AcceleratorRead);

    const int Nsimd = CComplex::Nsimd();
    typedef decltype(coalescedRead(in_v[0])) calcVector;
    typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

    int osites=Grid()->oSites();
    //    double flops = osites*Nsimd*nbasis*nbasis*8.0*geom.npoint;
    //    double bytes = osites*nbasis*nbasis*geom.npoint*sizeof(CComplex);
    double usecs =-usecond();

    accelerator_for(sss, Grid()->oSites()*nbasis, Nsimd, {
      int ss = sss/nbasis;
      int b  = sss%nbasis;
      calcComplex res = Zero();
      calcVector nbr;
      int ptype;
      StencilEntry *SE;

      int lane=acceleratorSIMTlane(Nsimd);
      for(int point=0;point<geom_.npoint;point++){

	SE=Stencil_v.GetEntry(ptype,point,ss);
	  
	if(SE->_is_local) { 
	  nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute,lane);
	} else {
	  nbr = coalescedRead(Stencil_v.CommBuf()[SE->_offset],lane);
	}
	acceleratorSynchronise();

	for(int bb=0;bb<nbasis;bb++) {
	  res = res + coalescedRead(Aview_p[point][ss](b,bb))*nbr(bb);
	}
      }
      coalescedWrite(out_v[ss](b),res,lane);
    });
    usecs +=usecond();

    /*
        std::cout << GridLogMessage << "\tHalo        " << comms_usec << " us" <<std::endl;
        std::cout << GridLogMessage << "\tMatrix      " << usecs << " us" <<std::endl;
        std::cout << GridLogMessage << "\t  mflop/s   " << flops/usecs<<std::endl;
        std::cout << GridLogMessage << "\t  MB/s      " << bytes/usecs<<std::endl;
    */
    vectorViewPointerClose(Aview_v, Aview_p);
  };

  void Mdag (const CoarseVector &in, CoarseVector &out)
  {
    if(hermitian) {
      // corresponds to Petrov-Galerkin coarsening
      return M(in,out);
    } else {
      // corresponds to Galerkin coarsening
      CoarseVector tmp(Grid());
      G5C(tmp, in); 
      M(tmp, out);
      G5C(out, out);
    }
  };
  void MdirComms(const CoarseVector &in)
  {
    SimpleCompressor<siteVector> compressor;
    Stencil.HaloExchange(in,compressor);
  }
  void MdirCalc(const CoarseVector &in, CoarseVector &out, int point)
  {
    conformable(_grid,in.Grid());
    conformable(_grid,out.Grid());

    vectorViewPointerOpen(Aview_v, Aview_p, A, AcceleratorRead);
    autoView( out_v , out, AcceleratorWrite);
    autoView( in_v  , in, AcceleratorRead);
    autoView( Stencil_v  , Stencil, AcceleratorRead);

    const int Nsimd = CComplex::Nsimd();
    typedef decltype(coalescedRead(in_v[0])) calcVector;
    typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

    accelerator_for(sss, Grid()->oSites()*nbasis, Nsimd, {
      int ss = sss/nbasis;
      int b  = sss%nbasis;
      calcComplex res = Zero();
      calcVector nbr;
      int ptype;
      StencilEntry *SE;

      int lane=acceleratorSIMTlane(Nsimd);
      SE=Stencil_v.GetEntry(ptype,point,ss);
	  
      if(SE->_is_local) { 
	nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute,lane);
      } else {
	nbr = coalescedRead(Stencil_v.CommBuf()[SE->_offset],lane);
      }
      acceleratorSynchronise();

      for(int bb=0;bb<nbasis;bb++) {
	res = res + coalescedRead(Aview_p[point][ss](b,bb))*nbr(bb);
      }
      coalescedWrite(out_v[ss](b),res,lane);
    });
    vectorViewPointerClose(Aview_v, Aview_p);
#if 0
    accelerator_for(ss,Grid()->oSites(),1,{

      siteVector res = Zero();
      siteVector nbr;
      int ptype;
      StencilEntry *SE;
      
      SE=Stencil_v.GetEntry(ptype,point,ss);
      
      if(SE->_is_local&&SE->_permute) {
	permute(nbr,in_v[SE->_offset],ptype);
      } else if(SE->_is_local) {
	nbr = in_v[SE->_offset];
      } else {
	nbr = Stencil_v.CommBuf()[SE->_offset];
      }
      acceleratorSynchronise();

      res = res + Aview_p[point][ss]*nbr;
      
      out_v[ss]=res;
    });
#endif
  }
  void MdirAll(const CoarseVector &in,std::vector<CoarseVector> &out)
  {
    this->MdirComms(in);
    int ndir=geom.npoint-1;
    if ((out.size()!=ndir)&&(out.size()!=ndir+1)) { 
      std::cout <<"MdirAll out size "<< out.size()<<std::endl;
      std::cout <<"MdirAll ndir "<< ndir<<std::endl;
      assert(0);
    }
    for(int p=0;p<ndir;p++){
      MdirCalc(in,out[p],p);
    }
  };
  void Mdir(const CoarseVector &in, CoarseVector &out, int dir, int disp){

    this->MdirComms(in);

    int ndim = in.Grid()->Nd();

    //////////////
    // 4D action like wilson
    // 0+ => 0
    // 0- => 4
    // 1+ => 1
    // 1- => 5
    // etc..
    //////////////
    // 5D action like DWF
    // 1+ => 0
    // 1- => 4
    // 2+ => 1
    // 2- => 5
    // etc..
    auto point = [dir, disp, ndim](){
      if(dir == 0 and disp == 0)
	return 8;
      else if ( ndim==4 ) { 
	return (1 - disp) / 2 * 4 + dir;
      } else { 
	return (1 - disp) / 2 * 4 + dir - 1;
      }
    }();

    MdirCalc(in,out,point);

  };

  void Mdiag(const CoarseVector &in, CoarseVector &out)
  {
    int point=geom.npoint-1;
    MdirCalc(in, out, point); // No comms
  };

  
 CoarsenedMatrix(GridCartesian &CoarseGrid, int hermitian_=0) 	: 

    _grid(&CoarseGrid),
    geom(CoarseGrid._ndimension),
    hermitian(hermitian_),
    Stencil(&CoarseGrid,geom.npoint,Even,geom.directions,geom.displacements,0),
      A(geom.npoint,&CoarseGrid)
  {
  };

  void CoarsenOperator(GridBase *FineGrid,LinearOperatorBase<Lattice<Fobj> > &linop,
		       Aggregation<Fobj,CComplex,nbasis> & Subspace)
  {
    prof_.Start("CoarsenOperator.Total");
    prof_.Start("CoarsenOperator.Misc");
    typedef Lattice<typename Fobj::tensor_reduced> FineComplexField;
    typedef typename Fobj::scalar_type scalar_type;

    FineComplexField one(FineGrid); one=scalar_type(1.0,0.0);
    FineComplexField zero(FineGrid); zero=scalar_type(0.0,0.0);

    FineComplexField omask(FineGrid);

    FineComplexField evenmask(FineGrid);
    FineComplexField oddmask(FineGrid); 

    FineField     phi(FineGrid);
    FineField     tmp(FineGrid);
    FineField     zz(FineGrid); zz=Zero();
    FineField    Mphi(FineGrid);
    FineField    Mphie(FineGrid);
    FineField    Mphio(FineGrid);
    std::vector<FineField>     Mphi_p(geom.npoint,FineGrid);

    Lattice<iScalar<vInteger> > coor (FineGrid);
    Lattice<iScalar<vInteger> > bcoor(FineGrid);
    Lattice<iScalar<vInteger> > bcb  (FineGrid); bcb = Zero();

    CoarseVector iProj(Grid()); 
    CoarseVector oProj(Grid()); 
    CoarseVector SelfProj(Grid()); 
    CoarseComplexField iZProj(Grid()); 
    CoarseComplexField oZProj(Grid()); 

    CoarseScalar InnerProd(Grid()); 

    Grid::Rework::CoarseningLookupTable ilut(Grid(), one);
    std::vector<Grid::Rework::CoarseningLookupTable> olut(geom.npoint);
    prof_.Stop("CoarsenOperator.Misc");

    // Orthogonalise the subblocks over the basis
    // blockOrthogonalise(InnerProd,Subspace.subspace); // NOTE: commented for comparisons to work, should be done outside anyway

    prof_.Start("CoarsenOperator.SetupMasks");
    // Compute the matrix elements of linop between this orthonormal
    // set of vectors.
    int self_stencil=-1;
    for(int p=0;p<geom.npoint;p++)
    { 
      int dir   = geom.directions[p];
      int disp  = geom.displacements[p];
      A[p]=Zero();
      if( geom.displacements[p]==0){
	self_stencil=p;
      }

      Integer block=(FineGrid->_rdimensions[dir])/(Grid()->_rdimensions[dir]);

      LatticeCoordinate(coor,dir);

      ///////////////////////////////////////////////////////
      // Work out even and odd block checkerboarding for fast diagonal term
      ///////////////////////////////////////////////////////
      if ( disp==1 ) {
	bcb   = bcb + div(coor,block);
      }
	
      if ( disp==0 ) {
	  omask= Zero();
      } else if ( disp==1 ) {
	omask = where(mod(coor,block)==(block-1),one,zero);
      } else if ( disp==-1 ) {
	omask = where(mod(coor,block)==(Integer)0,one,zero);
      }

      olut[p].populate(Grid(), omask);
    }
    evenmask = where(mod(bcb,2)==(Integer)0,one,zero);
    oddmask  = one-evenmask;

    assert(self_stencil!=-1);
    prof_.Stop("CoarsenOperator.SetupMasks");

    for(int i=0;i<nbasis;i++){

      prof_.Start("CoarsenOperator.ExtractVector");
      phi=Subspace.subspace[i];
      prof_.Stop("CoarsenOperator.ExtractVector");

      //      std::cout << GridLogMessage<< "CoarsenMatrix vector "<<i << std::endl;
      prof_.Start("CoarsenOperator.ApplyOpFirst");
      linop.OpDirAll(phi,Mphi_p);
      linop.OpDiag  (phi,Mphi_p[geom.npoint-1]);
      prof_.Stop("CoarsenOperator.ApplyOpFirst");

      for(int p=0;p<geom.npoint;p++){ 

	prof_.Start("CoarsenOperator.CopyVector");
	Mphi = Mphi_p[p];
	prof_.Stop("CoarsenOperator.CopyVector");

	int dir   = geom.directions[p];
	int disp  = geom.displacements[p];

	if (disp==-1) {

	  prof_.Start("CoarsenOperator.ProjectToSubspaceOuter");
	  Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct(oProj,Mphi,Subspace.subspace,olut[p]);
	  prof_.Stop("CoarsenOperator.ProjectToSubspaceOuter");

	  prof_.Start("CoarsenOperator.ConstructLinksProj");
	  autoView(oProj_v, oProj, AcceleratorRead);
	  autoView(A_p, A[p], AcceleratorWrite);
	  accelerator_for(ss, Grid()->oSites(), Fobj::Nsimd(),{
	    for(int j=0;j<nbasis;j++){
	      coalescedWrite(A_p[ss](j,i),oProj_v(ss)(j));
	    }
	  });
	  prof_.Stop("CoarsenOperator.ConstructLinksProj");
	}
      }

      ///////////////////////////////////////////
      // Faster alternate self coupling.. use hermiticity to save 2x
      ///////////////////////////////////////////
      {
	prof_.Start("CoarsenOperator.ApplyOpSecond");
	mult(tmp,phi,evenmask);  linop.Op(tmp,Mphie);
	mult(tmp,phi,oddmask );  linop.Op(tmp,Mphio);
	prof_.Stop("CoarsenOperator.ApplyOpSecond");

	prof_.Start("CoarsenOperator.AccumInner");
	{
	  autoView(tmp_, tmp, AcceleratorWrite);
	  autoView(evenmask_, evenmask, AcceleratorRead);
	  autoView(oddmask_,  oddmask, AcceleratorRead);
	  autoView(Mphie_,  Mphie, AcceleratorRead);
	  autoView(Mphio_,  Mphio, AcceleratorRead);
	  accelerator_for(ss, FineGrid->oSites(), Fobj::Nsimd(),{ 
	      coalescedWrite(tmp_[ss],evenmask_(ss)*Mphie_(ss) + oddmask_(ss)*Mphio_(ss));
	    });
	}
	prof_.Stop("CoarsenOperator.AccumInner");

	prof_.Start("CoarsenOperator.ProjectToSubspaceInner");
	Grid::UpstreamImprovedDirsaveLut::blockLutedInnerProduct(SelfProj,tmp,Subspace.subspace,ilut);
	prof_.Stop("CoarsenOperator.ProjectToSubspaceInner");

	prof_.Start("CoarsenOperator.ConstructLinksSelf");
	autoView(SelfProj_, SelfProj, AcceleratorRead);
	autoView(A_self, A[self_stencil], AcceleratorWrite);

	accelerator_for(ss, Grid()->oSites(), Fobj::Nsimd(),{
	  for(int j=0;j<nbasis;j++){
	    coalescedWrite(A_self[ss](j,i), SelfProj_(ss)(j));
	  }
	});
	prof_.Stop("CoarsenOperator.ConstructLinksSelf");

      }
    }
    if(hermitian) {
      std::cout << GridLogMessage << " ForceHermitian, new code "<<std::endl;
      ForceHermitian();
    } else {
      ConstructRemainingLinks();
    }
      // AssertHermitian();
      // ForceDiagonal();
    prof_.Stop("CoarsenOperator.Total");
  }

#if 0
    ///////////////////////////
    // test code worth preserving in if block
    ///////////////////////////
    std::cout<<GridLogMessage<< " Computed matrix elements "<< self_stencil <<std::endl;
    for(int p=0;p<geom.npoint;p++){
      std::cout<<GridLogMessage<< "A["<<p<<"]" << std::endl;
      std::cout<<GridLogMessage<< A[p] << std::endl;
    }
    std::cout<<GridLogMessage<< " picking by block0 "<< self_stencil <<std::endl;

    phi=Subspace.subspace[0];
    std::vector<int> bc(FineGrid->_ndimension,0);

    blockPick(Grid(),phi,tmp,bc);      // Pick out a block
    linop.Op(tmp,Mphi);                // Apply big dop
    blockProject(iProj,Mphi,Subspace.subspace); // project it and print it
    std::cout<<GridLogMessage<< " Computed matrix elements from block zero only "<<std::endl;
    std::cout<<GridLogMessage<< iProj <<std::endl;
    std::cout<<GridLogMessage<<"Computed Coarse Operator"<<std::endl;
#endif


  void ForceHermitian(void) {
    prof_.Start("CoarsenOperator.ForceHermitian");
    CoarseMatrix Diff  (Grid());
    for(int p=0;p<geom.npoint;p++){
      int dir   = geom.directions[p];
      int disp  = geom.displacements[p];
      if(disp==-1) {
	// Find the opposite link
	for(int pp=0;pp<geom.npoint;pp++){
	  int dirp   = geom.directions[pp];
	  int dispp  = geom.displacements[pp];
	  if ( (dirp==dir) && (dispp==1) ){
	    //	    Diff = adj(Cshift(A[p],dir,1)) - A[pp]; 
	    //	    std::cout << GridLogMessage<<" Replacing stencil leg "<<pp<<" with leg "<<p<< " diff "<<norm2(Diff) <<std::endl;
	    A[pp] = adj(Cshift(A[p],dir,1));
	  }
	}
      }
    }
    prof_.Stop("CoarsenOperator.ForceHermitian");
  }
  void AssertHermitian(void) {
    CoarseMatrix AA    (Grid());
    CoarseMatrix AAc   (Grid());
    CoarseMatrix Diff  (Grid());
    for(int d=0;d<4;d++){
	
      int dd=d+1;
      AAc = Cshift(A[2*d+1],dd,1);
      AA  = A[2*d];
	
      Diff = AA - adj(AAc);

      std::cout<<GridLogMessage<<"Norm diff dim "<<d<<" "<< norm2(Diff)<<std::endl;
      std::cout<<GridLogMessage<<"Norm dim "<<d<<" "<< norm2(AA)<<std::endl;
	  
    }
    Diff = A[8] - adj(A[8]);
    std::cout<<GridLogMessage<<"Norm diff local "<< norm2(Diff)<<std::endl;
    std::cout<<GridLogMessage<<"Norm local "<< norm2(A[8])<<std::endl;
  }
    
  void ConstructRemainingLinks(void) {
    prof_.Start("CoarsenOperator.ConstructLinksComm");
    for(int p=0;p<geom.npoint;p++){
      int dir   = geom.directions[p];
      int disp  = geom.displacements[p];
      if(disp==-1) {
        auto tmp   = closure(adj(A[p]));
        auto tmp_v = tmp.View();
        accelerator_for(ss, Grid()->oSites(), Fobj::Nsimd(), {
          Real factor;
          auto tmp_t = tmp_v(ss);
          for(int n1 = 0; n1 < nbasis; ++n1) {
            int k = n1/(nbasis/2);
            for(int n2 = 0; n2 < nbasis; ++n2) {
              int l = n2/(nbasis/2);
              factor        = ((k + l) % 2 == 1) ? -1. : 1.;
              tmp_t(n1, n2) = factor * tmp_t(n1, n2);
            }
          }
          coalescedWrite(tmp_v[ss], tmp_t);
        });
        // Find the opposite link
        for(int pp=0;pp<geom.npoint;pp++){
          int dirp   = geom.directions[pp];
          int dispp  = geom.displacements[pp];
          if ( (dirp==dir) && (dispp==1) ){
            A[pp] = Cshift(tmp,dir,1);
          }
        }
      }
    }
    prof_.Stop("CoarsenOperator.ConstructLinksComm");
  }

};

NAMESPACE_END(UpstreamImprovedDirsaveLut);
NAMESPACE_END(Grid);
#endif
