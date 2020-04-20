/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/CoarseningPolicyRework.h

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

#define INHERIT_COARSENING_POLICY_TYPES(Policy) \
  typedef typename Policy::SiteLinkField     SiteLinkField; \
  typedef typename Policy::SiteSpinor        SiteSpinor; \
  typedef typename Policy::SiteScalar        SiteScalar; \
  typedef typename Policy::SiteSpinVector    SiteSpinVector; \
  typedef typename Policy::LinkField         LinkField; \
  typedef typename Policy::FermionField      FermionField; \
  typedef typename Policy::ScalarField       ScalarField; \
  typedef typename Policy::SpinVectorField   SpinVectorField; \
  typedef typename Policy::FineSiteSpinor    FineSiteSpinor; \
  typedef typename Policy::FineSiteScalar    FineSiteScalar; \
  typedef typename Policy::FineFermionField  FineFermionField; \
  typedef typename Policy::FineScalarField   FineScalarField; \
  typedef typename Policy::IntergridOperator IntergridOperator; \
  typedef typename Policy::Simd              Simd;

#define INHERIT_COARSENING_POLICY_VARIABLES(Policy) \
  using Policy::Ns_f; \
  using Policy::Ns_c; \
  using Policy::Nc_f; \
  using Policy::Nc_c; \
  using Policy::Ns_b;

template<class _FineFermionField, int _Nbasis, int _Ncoarse_spins>
class CoarseningPolicy {
public:
  /////////////////////////////////////////////
  // Static variables
  /////////////////////////////////////////////

  // these get the info I need at compile time instead of run time like indexRank does
  static constexpr int Nl_f = GridTypeMapper<typename getVectorType<_FineFermionField>::type>::Dimension(LorentzIndex);
  static constexpr int Ns_f = GridTypeMapper<typename getVectorType<_FineFermionField>::type>::Dimension(SpinIndex);
  static constexpr int Nc_f = GridTypeMapper<typename getVectorType<_FineFermionField>::type>::Dimension(ColourIndex);
  static constexpr int Ns_c = _Ncoarse_spins;
  static constexpr int Nc_c = _Nbasis;

  static_assert(Ns_f >= Ns_c, "Number of spin dofs incorrect");
  static_assert(Nl_f == 1,    "Number of lorentz dofs incorrect");

  static constexpr int Ns_b = Ns_f/Ns_c; // spin blocking

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  template<typename vtype> using iImplLinkField  = iScalar<iMatrix<iMatrix<vtype, Nc_c>, Ns_c>>;
  template<typename vtype> using iImplSpinor     = iScalar<iVector<iVector<vtype, Nc_c>, Ns_c>>;
  template<typename vtype> using iImplSpinorSplit = iVector<iVector<iVector<vtype, Nc_c>, Ns_c>, Ns_c>;
  template<typename vtype> using iImplScalar     = iScalar<iScalar<iScalar<vtype>>>;
  template<typename vtype> using iImplSpinVector = iScalar<iVector<iScalar<vtype>, Ns_c>>;

  typedef typename _FineFermionField::vector_type Simd; // use same SIMD type as on fine grid

  typedef iImplLinkField<Simd>  SiteLinkField;
  typedef iImplSpinor<Simd>     SiteSpinor;
  typedef iImplSpinorSplit<Simd>     SiteSpinorSplit;
  typedef iImplScalar<Simd>     SiteScalar;
  typedef iImplSpinVector<Simd> SiteSpinVector;

  // Needs to be called FermionField like in the Dirac operators,
  // so that we can use it for coarsening just like the Dirac operators
  typedef Lattice<SiteLinkField>  LinkField;
  typedef Lattice<SiteSpinor>     FermionField;
  typedef Lattice<SiteSpinorSplit>     FermionFieldSplit;
  typedef Lattice<SiteScalar>     ScalarField;
  typedef Lattice<SiteSpinVector> SpinVectorField;

  typedef typename getVectorType<_FineFermionField>::type FineSiteSpinor;
  typedef typename FineSiteSpinor::tensor_reduced         FineSiteScalar;

  typedef _FineFermionField       FineFermionField;
  typedef Lattice<iVector<iVector<iVector<Simd, Nc_f>, Ns_f>, Ns_c>> FineFermionFieldSplit;
  typedef Lattice<FineSiteScalar> FineScalarField;

  typedef std::vector<FineFermionField> IntergridOperator; // this can then freely be changed if we would require it to
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
