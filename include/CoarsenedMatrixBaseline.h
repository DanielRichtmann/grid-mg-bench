/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/algorithms/CoarsenedMatrixBaseline.h

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
#ifndef  GRID_ALGORITHM_COARSENED_MATRIX_BASELINE_H
#define  GRID_ALGORITHM_COARSENED_MATRIX_BASELINE_H

#include <Grid/qcd/QCD.h> // needed for Dagger(Yes|No), Inverse(Yes|No)

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(Baseline);

class Geometry {
  //    int dimension;
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
    std::cout <<std::endl;
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
  int Checkerboard(void){return checkerboard;}
  Aggregation(GridBase *_CoarseGrid,GridBase *_FineGrid,int _checkerboard) :
    CoarseGrid(_CoarseGrid),
    FineGrid(_FineGrid),
    subspace(nbasis,_FineGrid),
    checkerboard(_checkerboard)
  {
  };

  std::vector<FineField> &Subspace() {
    return subspace;
  }
  
  void Orthogonalise(int checkOrthogonality = 1, int passes = 2){
    CoarseScalar InnerProd(CoarseGrid); 
    for(int n = 0; n < passes; ++n) {
      std::cout << GridLogMessage <<" Gramm-Schmidt pass "<<n+1<<std::endl;
      blockOrthogonalise(InnerProd, subspace);
    }
    if(checkOrthogonality) CheckOrthogonal();
  }
  void CheckOrthogonal(void){
    CoarseVector iProj(CoarseGrid);
    CoarseVector eProj(CoarseGrid);
    for(int i=0;i<nbasis;i++){
      ProjectToSubspace(iProj,subspace[i]);
      eProj=Zero(); 
      auto eProj_v = eProj.View();
      thread_for(ss, CoarseGrid->oSites(),{
	eProj_v[ss](i)=CComplex(1.0);
      });
      eProj=eProj - iProj;
      std::cout<<GridLogMessage<<"Orthog check error "<<i<<" " << norm2(eProj)<<std::endl;
    }
    std::cout<<GridLogMessage <<"CheckOrthog done"<<std::endl;
  }
  void ProjectToSubspace(CoarseVector &CoarseVec,const FineField &FineVec){
    blockProject(CoarseVec,FineVec,subspace);
  }
  void PromoteFromSubspace(const CoarseVector &CoarseVec,FineField &FineVec){
    FineVec.Checkerboard() = subspace[0].Checkerboard();
    blockPromote(CoarseVec,FineVec,subspace);
  }
  void CreateSubspaceRandom(GridParallelRNG &RNG){
    for(int i=0;i<nbasis;i++){
      random(RNG,subspace[i]);
    }
  }

  /*
    virtual void CreateSubspaceLanczos(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis)
    {
    // Run a Lanczos with sloppy convergence
    const int Nstop = nn;
    const int Nk = nn+20;
    const int Np = nn+20;
    const int Nm = Nk+Np;
    const int MaxIt= 10000;
    RealD resid = 1.0e-3;

    Chebyshev<FineField> Cheb(0.5,64.0,21);
    ImplicitlyRestartedLanczos<FineField> IRL(hermop,Cheb,Nstop,Nk,Nm,resid,MaxIt);
    //	IRL.lock = 1;

    FineField noise(FineGrid); gaussian(RNG,noise);
    FineField tmp(FineGrid);
    std::vector<RealD>     eval(Nm);
    std::vector<FineField> evec(Nm,FineGrid);

    int Nconv;
    IRL.calc(eval,evec,
    noise,
    Nconv);

    // pull back nn vectors
    for(int b=0;b<nn;b++){

    subspace[b]   = evec[b];

    std::cout << GridLogMessage <<"subspace["<<b<<"] = "<<norm2(subspace[b])<<std::endl;

    hermop.Op(subspace[b],tmp);
    std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(tmp)<<std::endl;

    noise = tmp -  sqrt(eval[b])*subspace[b] ;

    std::cout<<GridLogMessage << " lambda_"<<b<<" = "<< eval[b] <<"  ;  [ M - Lambda ]_"<<b<<" vec_"<<b<<"  = " <<norm2(noise)<<std::endl;

    noise = tmp +  eval[b]*subspace[b] ;

    std::cout<<GridLogMessage << " lambda_"<<b<<" = "<< eval[b] <<"  ;  [ M - Lambda ]_"<<b<<" vec_"<<b<<"  = " <<norm2(noise)<<std::endl;

    }
    Orthogonalise();
    for(int b=0;b<nn;b++){
    std::cout << GridLogMessage <<"subspace["<<b<<"] = "<<norm2(subspace[b])<<std::endl;
    }
    }
  */
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

  virtual void CreateSubspaceChebyshev(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis) {

    RealD scale;


    Chebyshev<FineField> Cheb(0.1,64.0,900);

    FineField noise(FineGrid);
    FineField Mn(FineGrid);

    for(int b=0;b<nn;b++){

      gaussian(RNG,noise);
      scale = std::pow(norm2(noise),-0.5);
      noise=noise*scale;

      hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise   ["<<b<<"] <n|MdagM|n> "<<norm2(Mn)<<std::endl;

      Cheb(hermop,noise,Mn);

      scale = std::pow(norm2(Mn),-0.5);
      Mn=Mn*scale;
      subspace[b]   = Mn;

      hermop.Op(Mn,noise); std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(noise)<<std::endl;

    }

    Orthogonalise();

  }

};
// Fine Object == (per site) type of fine field
// nbasis      == number of deflation vectors
template<class Fobj,class CComplex,int nbasis>
class CoarsenedMatrix : public CheckerBoardedSparseMatrixBase<Lattice<iVector<CComplex,nbasis > > >
                      , public Rework::Profileable {
public:

  typedef iVector<CComplex,nbasis >             siteVector;
  typedef Lattice<siteVector>                 CoarseVector;
  typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;

  typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
  typedef Lattice<Fobj >        FineField;

  ////////////////////
  // Data members
  ////////////////////
  Geometry         geom;
  GridBase *       _grid;
  GridBase *       _cbgrid;
  int hermitian;

  CartesianStencil<siteVector,siteVector,int> Stencil; 
  CartesianStencil<siteVector,siteVector,int> StencilEven;
  CartesianStencil<siteVector,siteVector,int> StencilOdd;

  std::vector<CoarseMatrix> A;
  std::vector<CoarseMatrix> AEven;
  std::vector<CoarseMatrix> AOdd;

  CoarseMatrix SelfStencilLinkInv;
  CoarseMatrix SelfStencilLinkInvEven;
  CoarseMatrix SelfStencilLinkInvOdd;

  ///////////////////////
  // Interface
  ///////////////////////
  GridBase * Grid(void)         { return _grid; };   // this is all the linalg routines need to know
  GridBase * RedBlackGrid(void) { return _cbgrid; }

  void M (const CoarseVector &in, CoarseVector &out){

    conformable(_grid,in.Grid());
    conformable(in.Grid(),out.Grid());

    SimpleCompressor<siteVector> compressor;
    Stencil.HaloExchange(in,compressor);
    auto in_v = in.View();
    auto out_v = out.View();
    thread_for(ss,Grid()->oSites(),{
      siteVector res = Zero();
      siteVector nbr;
      int ptype;
      StencilEntry *SE;
      for(int point=0;point<geom.npoint;point++){

	SE=Stencil.GetEntry(ptype,point,ss);

	if(SE->_is_local&&SE->_permute) {
	  permute(nbr,in_v[SE->_offset],ptype);
	} else if(SE->_is_local) {
	  nbr = in_v[SE->_offset];
	} else {
	  nbr = Stencil.CommBuf()[SE->_offset];
	}
	auto A_point = A[point].View();
	res = res + A_point[ss]*nbr;
      }
      vstream(out_v[ss],res);
    });
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

  void Mdir(const CoarseVector &in, CoarseVector &out, int dir, int disp){
    DhopDir(in, out, dir, disp);
  };

  void DhopDir(const CoarseVector &in, CoarseVector &out, int dir, int disp) {
    conformable(_grid, in.Grid()); // verifies full grid
    conformable(in.Grid(), out.Grid());

    SimpleCompressor<siteVector> compressor;
    Stencil.HaloExchange(in,compressor);

    auto point = [dir, disp](){
      if(dir == 0 and disp == 0)
	return 8;
      else
	return (4 * dir + 1 - disp) / 2;
    }();

    auto out_v = out.View();
    auto in_v  = in.View();
    thread_for(ss,Grid()->oSites(),{
      siteVector res = Zero();
      siteVector nbr;
      int ptype;
      StencilEntry *SE;

      SE=Stencil.GetEntry(ptype,point,ss);

      if(SE->_is_local&&SE->_permute) {
	permute(nbr,in_v[SE->_offset],ptype);
      } else if(SE->_is_local) {
	nbr = in_v[SE->_offset];
      } else {
	nbr = Stencil.CommBuf()[SE->_offset];
      }

      auto A_point = A[point].View();
      res = res + A_point[ss]*nbr;

      vstream(out_v[ss],res);
    });
  };

  void Mdiag(const CoarseVector &in, CoarseVector &out){
    Mooee(in, out); // just like for fermion operators
  };

  void Meooe(const CoarseVector &in, CoarseVector &out) {
    if(in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerNo);
    } else {
      DhopOE(in, out, DaggerNo);
    }
  }

  void MeooeDag(const CoarseVector &in, CoarseVector &out) {
    if(in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerYes);
    } else {
      DhopOE(in, out, DaggerYes);
    }
  }

  void Mooee(const CoarseVector &in, CoarseVector &out) {
    MooeeInternal(in, out, DaggerNo, InverseNo);
  }

  void MooeeInv(const CoarseVector &in, CoarseVector &out) {
    MooeeInternal(in, out, DaggerNo, InverseYes);
  }

  void MooeeDag(const CoarseVector &in, CoarseVector &out) {
    MooeeInternal(in, out, DaggerYes, InverseNo);
  }

  void MooeeInvDag(const CoarseVector &in, CoarseVector &out) {
    MooeeInternal(in, out, DaggerYes, InverseYes);
  }

  void Dhop(const CoarseVector &in, CoarseVector &out, int dag) {
    conformable(in.Grid(), _grid); // verifies full grid
    conformable(in.Grid(), out.Grid());

    out.Checkerboard() = in.Checkerboard();

    DhopInternal(Stencil, A, in, out, dag);
  }

  void DhopOE(const CoarseVector &in, CoarseVector &out, int dag) {
    conformable(in.Grid(), _cbgrid);   // verifies half grid
    conformable(in.Grid(), out.Grid()); // drops the cb check

    assert(in.Checkerboard() == Even);
    out.Checkerboard() = Odd;

    DhopInternal(StencilEven, AOdd, in, out, dag);
  }

  void DhopEO(const CoarseVector &in, CoarseVector &out, int dag) {
    conformable(in.Grid(), _cbgrid);   // verifies half grid
    conformable(in.Grid(), out.Grid()); // drops the cb check

    assert(in.Checkerboard() == Odd);
    out.Checkerboard() = Even;

    DhopInternal(StencilOdd, AEven, in, out, dag);
  }

  void MooeeInternal(const CoarseVector &in, CoarseVector &out, int dag, int inv) {
    // Implementation along the lines of clover
    out.Checkerboard() = in.Checkerboard();
    assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);
    CoarseMatrix *SelfStencil = nullptr;

    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd)
        SelfStencil = (inv) ? &SelfStencilLinkInvOdd : &AOdd[geom.npoint-1];
      else
        SelfStencil = (inv) ? &SelfStencilLinkInvEven : &AEven[geom.npoint-1];
    } else {
      SelfStencil = (inv) ? &SelfStencilLinkInv : &A[geom.npoint-1];
    }

    assert(SelfStencil != nullptr);

    if(dag)
      out = adj(*SelfStencil) * in;
    else
      out = *SelfStencil * in;
  }

  void DhopInternal(CartesianStencil<siteVector, siteVector, int> &st, std::vector<CoarseMatrix> &Y, const CoarseVector &in, CoarseVector &out, int dag) {
    if (dag) {
      CoarseVector tmp(in.Grid());   tmp.Checkerboard() = in.Checkerboard();
      CoarseVector tmp2(out.Grid()); tmp2.Checkerboard() = out.Checkerboard();
      G5C(tmp, in);

      SimpleCompressor<siteVector> compressor;
      st.HaloExchange(tmp, compressor);

      auto tmp_v = tmp.View();
      auto tmp2_v = tmp2.View();
      thread_for(ss, tmp.Grid()->oSites(), {
        siteVector    res = Zero();
        siteVector    nbr;
        int           ptype;
        StencilEntry *SE;
        for(int point = 0; point < geom.npoint; point++) {
          if(point != geom.npoint-1) {
            SE = st.GetEntry(ptype, point, ss);

            if(SE->_is_local && SE->_permute) {
              permute(nbr, tmp_v[SE->_offset], ptype);
            } else if(SE->_is_local) {
              nbr = tmp_v[SE->_offset];
            } else {
              nbr = st.CommBuf()[SE->_offset];
            }
            auto Y_point = Y[point].View();
            res = res + Y_point[ss] * nbr;
          }
        }
        vstream(tmp2_v[ss], res);
      });
      G5C(out, tmp2);
    } else {
      SimpleCompressor<siteVector> compressor;
      st.HaloExchange(in, compressor);

      auto in_v = in.View();
      auto out_v = out.View();
      thread_for(ss, in.Grid()->oSites(), {
        siteVector    res = Zero();
        siteVector    nbr;
        int           ptype;
        StencilEntry *SE;
        for(int point = 0; point < geom.npoint; point++) {
          if(point != geom.npoint-1) {
            SE = st.GetEntry(ptype, point, ss);

            if(SE->_is_local && SE->_permute) {
              permute(nbr, in_v[SE->_offset], ptype);
            } else if(SE->_is_local) {
              nbr = in_v[SE->_offset];
            } else {
              nbr = st.CommBuf()[SE->_offset];
            }
            auto Y_point = Y[point].View();
            res = res + Y_point[ss] * nbr;
          }
        }
        vstream(out_v[ss], res);
      });
    }
  }


 CoarsenedMatrix(GridCartesian &CoarseGrid, GridRedBlackCartesian &CoarseRBGrid, int hermitian_=0) 	:

    _grid(&CoarseGrid),
    _cbgrid(&CoarseRBGrid),
    geom(CoarseGrid._ndimension),
    hermitian(hermitian_),
    Stencil(&CoarseGrid,geom.npoint,Even,geom.directions,geom.displacements,0),
    StencilEven(&CoarseRBGrid,geom.npoint,Even,geom.directions,geom.displacements,0),
    StencilOdd(&CoarseRBGrid,geom.npoint,Odd,geom.directions,geom.displacements,0),
    A(geom.npoint,&CoarseGrid),
    AEven(geom.npoint,&CoarseRBGrid), // TODO: What do we do with the last stencil point here?
    AOdd(geom.npoint,&CoarseRBGrid),
    SelfStencilLinkInv(&CoarseGrid),
    SelfStencilLinkInvEven(&CoarseRBGrid),
    SelfStencilLinkInvOdd(&CoarseRBGrid)
  {
  }; // TODO: What do we do with the last stencil point here?

  void CoarsenOperator(GridBase *FineGrid,LinearOperatorBase<Lattice<Fobj> > &linop,
		       Aggregation<Fobj,CComplex,nbasis> & Subspace){
    prof_.Start("CoarsenOperator.Total");
    prof_.Start("CoarsenOperator.Misc");

    FineField iblock(FineGrid); // contributions from within this block
    FineField oblock(FineGrid); // contributions from outwith this block

    FineField     phi(FineGrid);
    FineField     tmp(FineGrid);
    FineField     zz(FineGrid); zz=Zero();
    FineField    Mphi(FineGrid);

    Lattice<iScalar<vInteger> > coor(FineGrid);

    CoarseVector iProj(Grid());
    CoarseVector oProj(Grid());
    CoarseScalar InnerProd(Grid());

    // Orthogonalise the subblocks over the basis
    // Subspace.Orthogonalise(1); // NOTE: commented for comparisons to work, should be done the outside anyway

    // Compute the matrix elements of linop between this orthonormal
    // set of vectors.
    int self_stencil=-1;
    for(int p=0;p<geom.npoint;p++){
      A[p]=Zero();
      if( geom.displacements[p]==0){
	self_stencil=p;
      }
    }
    assert(self_stencil!=-1);
    prof_.Stop("CoarsenOperator.Misc");

    for(int i=0;i<nbasis;i++){
      prof_.Start("CoarsenOperator.Copy");
      phi=Subspace.subspace[i];
      prof_.Stop("CoarsenOperator.Copy");

      std::cout<<GridLogMessage<<"("<<i<<").."<<std::endl;

      for(int p=0;p<geom.npoint;p++){

        prof_.Start("CoarsenOperator.LatticeCoordinate");
	int dir   = geom.directions[p];
	int disp  = geom.displacements[p];

	Integer block=(FineGrid->_rdimensions[dir])/(Grid()->_rdimensions[dir]);

	LatticeCoordinate(coor,dir);
        prof_.Stop("CoarsenOperator.LatticeCoordinate");

        prof_.Start("CoarsenOperator.ApplyOp");
	if ( disp==0 ){
	  linop.OpDiag(phi,Mphi);
	}
	else  {
	  linop.OpDir(phi,Mphi,dir,disp);
	}
        prof_.Stop("CoarsenOperator.ApplyOp");

	////////////////////////////////////////////////////////////////////////
	// Pick out contributions coming from this cell and neighbour cell
	////////////////////////////////////////////////////////////////////////
        prof_.Start("CoarsenOperator.PickBlocks");
	if ( disp==0 ) {
	  iblock = Mphi;
	  oblock = Zero();
	} else if ( disp==1 ) {
	  oblock = where(mod(coor,block)==(block-1),Mphi,zz);
	  iblock = where(mod(coor,block)!=(block-1),Mphi,zz);
	} else if ( disp==-1 ) {
	  oblock = where(mod(coor,block)==(Integer)0,Mphi,zz);
	  iblock = where(mod(coor,block)!=(Integer)0,Mphi,zz);
	} else {
	  assert(0);
	}
        prof_.Stop("CoarsenOperator.PickBlocks");

        prof_.Start("CoarsenOperator.ProjectToSubspace");
	Subspace.ProjectToSubspace(iProj,iblock);
	Subspace.ProjectToSubspace(oProj,oblock);
        prof_.Stop("CoarsenOperator.ProjectToSubspace", 2);
	//	  blockProject(iProj,iblock,Subspace.subspace);
	//	  blockProject(oProj,oblock,Subspace.subspace);
        prof_.Start("CoarsenOperator.ConstructLinksFull");
	auto iProj_v = iProj.View() ;
	auto oProj_v = oProj.View() ;
	auto A_p     =  A[p].View();
	auto A_self  = A[self_stencil].View();
	thread_for(ss, Grid()->oSites(),{
	  for(int j=0;j<nbasis;j++){
	    if( disp!= 0 ) {
	      A_p[ss](j,i) = oProj_v[ss](j);
	    }
	    A_self[ss](j,i) =	A_self[ss](j,i) + iProj_v[ss](j);
	  }
	});
        prof_.Stop("CoarsenOperator.ConstructLinksFull");
      }
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
      //      ForceHermitian();
      // AssertHermitian();
      // ForceDiagonal();

    InvertSelfStencilPoint();
    FillHalfCbs();
    prof_.Stop("CoarsenOperator.Total");
  }

  void ForceHermitian(void) {
    for(int d=0;d<4;d++){
      int dd=d+1;
      A[2*d] = adj(Cshift(A[2*d+1],dd,1));
    }
    //      A[8] = 0.5*(A[8] + adj(A[8]));
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
    
  void InvertSelfStencilPoint() {
    prof_.Start("CoarsenOperator.InvertSelfStencilPoint");
    int localVolume = Grid()->lSites();

    using scalar_object = typename iMatrix<CComplex, nbasis>::scalar_object;

    thread_for(site, localVolume, {
      Eigen::MatrixXcd EigenSelfStencil    = Eigen::MatrixXcd::Zero(nbasis, nbasis);
      Eigen::MatrixXcd EigenInvSelfStencil = Eigen::MatrixXcd::Zero(nbasis, nbasis);

      scalar_object SelfStencilLink    = Zero();
      scalar_object InvSelfStencilLink = Zero();

      Coordinate lcoor;

      Grid()->LocalIndexToLocalCoor(site, lcoor);
      EigenSelfStencil = Eigen::MatrixXcd::Zero(nbasis, nbasis);
      peekLocalSite(SelfStencilLink, A[geom.npoint-1], lcoor);
      InvSelfStencilLink = Zero();

      for (int i = 0; i < nbasis; ++i)
        for (int j = 0; j < nbasis; ++j)
          EigenSelfStencil(i, j) = static_cast<ComplexD>(TensorRemove(SelfStencilLink(i, j)));

      EigenInvSelfStencil = EigenSelfStencil.inverse();

      for(int i = 0; i < nbasis; ++i)
        for(int j = 0; j < nbasis; ++j)
          InvSelfStencilLink(i, j) = EigenInvSelfStencil(i, j);

      pokeLocalSite(InvSelfStencilLink, SelfStencilLinkInv, lcoor);
    });
    prof_.Stop("CoarsenOperator.InvertSelfStencilPoint");
  }

  void FillHalfCbs() {
    prof_.Start("CoarsenOperator.FillHalfCbs");
    for(int p = 0; p < geom.npoint; p++) {
      pickCheckerboard(Even, AEven[p], A[p]);
      pickCheckerboard(Odd, AOdd[p], A[p]);
    }
    pickCheckerboard(Even, SelfStencilLinkInvEven, SelfStencilLinkInv);
    pickCheckerboard(Odd, SelfStencilLinkInvOdd, SelfStencilLinkInv);
    prof_.Stop("CoarsenOperator.FillHalfCbs");
  }

  void MdirAll(const CoarseVector &in, std::vector<CoarseVector> &out) {}
};

NAMESPACE_END(Baseline);
NAMESPACE_END(Grid);
#endif
