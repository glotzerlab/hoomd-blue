// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "HPMCPrecisionSetup.h"

#ifndef __SPHINXOVERLAP__H__
#define __SPHINXOVERLAP__H__

namespace hpmc
{

/*! \file SphinxOverlap.h
    \brief Implements Sphere Intersections Overlap (Beth's version)
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace detail
{

// introducing a math function that I need and didn't find in VectorMath.h
// this function calculates the square of a vector

DEVICE inline OverlapReal norm2(const vec3<OverlapReal>& v)
    {
    return dot(v,v);
    }

//! Sphinx Volume of intersection algorithm for overlap detection
/*! \tparam D Distance^2 between every pair of spheres
    \param n Number of spheres

    This algorithm was developed by Elizabeth R Chen and is available at this address:
    https://dl.dropboxusercontent.com/u/38305351/SPHINX_overlap.pdf
    This version was introduced in HPMC by Khalid Ahmed with modifications.

    \ingroup sphinx
*/

//! Yes, this overlap check would also work up to DIM=6, but we have no use for it
#define DIM 3

#define EPS 1e-12


DEVICE inline OverlapReal ang4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
        OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
        OverlapReal cd,OverlapReal ce,OverlapReal cf,
        OverlapReal de,OverlapReal df,
        OverlapReal ef);
DEVICE inline OverlapReal ang5(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,
                OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,
                OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,
                OverlapReal de,OverlapReal df,OverlapReal dg,
                OverlapReal ef,OverlapReal eg,
                OverlapReal fg);
DEVICE inline bool sep2(bool convex,
              OverlapReal as,OverlapReal bs,
              OverlapReal ar,OverlapReal br,
              OverlapReal ab);
DEVICE inline  bool seq2(OverlapReal as,OverlapReal bs,
              OverlapReal ar,OverlapReal br,
              OverlapReal ab);




DEVICE inline  OverlapReal vok4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
                OverlapReal bc,OverlapReal bd,OverlapReal be,
                OverlapReal cd,OverlapReal ce,
                OverlapReal de)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        OverlapReal abce = ab*ce,acbe = ac*be,aebc = ae*bc;
        OverlapReal abde = ab*de,adbe = ad*be,aebd = ae*bd;
        OverlapReal acde = ac*de,adce = ad*ce,aecd = ae*cd;
        OverlapReal bcde = bc*de,bdce = bd*ce,becd = be*cd;

        OverlapReal Qabcd = acbd+adbc-abcd,Qacbd = abcd+adbc-acbd,Qadbc = abcd+acbd-adbc;
        OverlapReal Qabce = acbe+aebc-abce,Qacbe = abce+aebc-acbe,Qaebc = abce+acbe-aebc;
        OverlapReal Qabde = adbe+aebd-abde,Qadbe = abde+aebd-adbe,Qaebd = abde+adbe-aebd;
        OverlapReal Qacde = adce+aecd-acde,Qadce = acde+aecd-adce,Qaecd = acde+adce-aecd;
        OverlapReal Qbcde = bdce+becd-bcde,Qbdce = bcde+becd-bdce,Qbecd = bcde+bdce-becd;

        return
        +abcd*(Qabce+Qabde+Qaecd+Qbecd-Qabcd-4*(ae*be+ce*de))
        +acbd*(Qacbe+Qaebd+Qacde+Qbdce-Qacbd-4*(ae*ce+be*de))
        +adbc*(Qaebc+Qadbe+Qadce+Qbcde-Qadbc-4*(ae*de+be*ce))
        +abce*(Qabcd+Qabde+Qadce+Qbdce-Qabce-4*ad*bd)
        +acbe*(Qacbd+Qadbe+Qacde+Qbecd-Qacbe-4*ad*cd)
        +aebc*(Qadbc+Qaebd+Qaecd+Qbcde-Qaebc-4*bd*cd)
        +abde*(Qabcd+Qabce+Qacde+Qbcde-Qabde-4*ac*bc)
        +adbe*(Qadbc+Qacbe+Qadce+Qbecd-Qadbe)
        +aebd*(Qacbd+Qaebc+Qaecd+Qbdce-Qaebd)
        +acde*(Qacbd+Qacbe+Qabde+Qbcde-Qacde)
        +adce*(Qadbc+Qabce+Qadbe+Qbdce-Qadce)
        +aecd*(Qabcd+Qaebc+Qaebd+Qbecd-Qaecd)
        +bcde*(Qadbc+Qaebc+Qabde+Qacde-Qbcde)
        +bdce*(Qacbd+Qabce+Qaebd+Qadce-Qbdce)
        +becd*(Qabcd+Qacbe+Qadbe+Qaecd-Qbecd);
    }

DEVICE inline bool vok5(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
              OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
              OverlapReal cd,OverlapReal ce,OverlapReal cf,
              OverlapReal de,OverlapReal df,
              OverlapReal ef)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        OverlapReal abce = ab*ce,acbe = ac*be,aebc = ae*bc;
        OverlapReal abcf = ab*cf,acbf = ac*bf,afbc = af*bc;
        OverlapReal abde = ab*de,adbe = ad*be,aebd = ae*bd;
        OverlapReal abdf = ab*df,adbf = ad*bf,afbd = af*bd;
        OverlapReal abef = ab*ef,aebf = ae*bf,afbe = af*be;
        OverlapReal acde = ac*de,adce = ad*ce,aecd = ae*cd;
        OverlapReal acdf = ac*df,adcf = ad*cf,afcd = af*cd;
        OverlapReal acef = ac*ef,aecf = ae*cf,afce = af*ce;
        OverlapReal adef = ad*ef,aedf = ae*df,afde = af*de;
        OverlapReal bcde = bc*de,bdce = bd*ce,becd = be*cd;
        OverlapReal bcdf = bc*df,bdcf = bd*cf,bfcd = bf*cd;
        OverlapReal bcef = bc*ef,becf = be*cf,bfce = bf*ce;
        OverlapReal bdef = bd*ef,bedf = be*df,bfde = bf*de;
        OverlapReal cdef = cd*ef,cedf = ce*df,cfde = cf*de;

        OverlapReal Qabcd = acbd+adbc-abcd,Qacbd = abcd+adbc-acbd,Qadbc = abcd+acbd-adbc;
        OverlapReal Qabce = acbe+aebc-abce,Qacbe = abce+aebc-acbe,Qaebc = abce+acbe-aebc;
        OverlapReal Qabcf = acbf+afbc-abcf,Qacbf = abcf+afbc-acbf,Qafbc = abcf+acbf-afbc;
        OverlapReal Qabde = adbe+aebd-abde,Qadbe = abde+aebd-adbe,Qaebd = abde+adbe-aebd;
        OverlapReal Qabdf = adbf+afbd-abdf,Qadbf = abdf+afbd-adbf,Qafbd = abdf+adbf-afbd;
        OverlapReal Qabef = aebf+afbe-abef,Qaebf = abef+afbe-aebf,Qafbe = abef+aebf-afbe;
        OverlapReal Qacde = adce+aecd-acde,Qadce = acde+aecd-adce,Qaecd = acde+adce-aecd;
        OverlapReal Qacdf = adcf+afcd-acdf,Qadcf = acdf+afcd-adcf,Qafcd = acdf+adcf-afcd;
        OverlapReal Qacef = aecf+afce-acef,Qaecf = acef+afce-aecf,Qafce = acef+aecf-afce;
        OverlapReal Qadef = aedf+afde-adef,Qaedf = adef+afde-aedf,Qafde = adef+aedf-afde;
        OverlapReal Qbcde = bdce+becd-bcde,Qbdce = bcde+becd-bdce,Qbecd = bcde+bdce-becd;
        OverlapReal Qbcdf = bdcf+bfcd-bcdf,Qbdcf = bcdf+bfcd-bdcf,Qbfcd = bcdf+bdcf-bfcd;
        OverlapReal Qbcef = becf+bfce-bcef,Qbecf = bcef+bfce-becf,Qbfce = bcef+becf-bfce;
        OverlapReal Qbdef = bedf+bfde-bdef,Qbedf = bdef+bfde-bedf,Qbfde = bdef+bedf-bfde;
        OverlapReal Qcdef = cedf+cfde-cdef,Qcedf = cdef+cfde-cedf,Qcfde = cdef+cedf-cfde;

        return
        +ab*cd*ef*(-Qabcd+Qabce+Qabcf+Qabde+Qabdf-Qabef+Qaecd+Qafcd+Qacef+Qadef+Qbecd+Qbfcd+Qbcef+Qbdef-Qcdef)
        +ab*ce*df*(+Qabcd-Qabce+Qabcf+Qabde-Qabdf+Qabef+Qadce+Qacdf+Qafce+Qaedf+Qbdce+Qbcdf+Qbfce+Qbedf-Qcedf)
        +ab*cf*de*(+Qabcd+Qabce-Qabcf-Qabde+Qabdf+Qabef+Qacde+Qadcf+Qaecf+Qafde+Qbcde+Qbdcf+Qbecf+Qbfde-Qcfde)
        +ac*bd*ef*(-Qacbd+Qacbe+Qacbf+Qaebd+Qafbd+Qabef+Qacde+Qacdf-Qacef+Qadef+Qbdce+Qbdcf+Qbcef-Qbdef+Qcdef)
        +ac*be*df*(+Qacbd-Qacbe+Qacbf+Qadbe+Qabdf+Qafbe+Qacde-Qacdf+Qacef+Qaedf+Qbecd+Qbcdf+Qbecf-Qbedf+Qcedf)
        +ac*bf*de*(+Qacbd+Qacbe-Qacbf+Qabde+Qadbf+Qaebf-Qacde+Qacdf+Qacef+Qafde+Qbcde+Qbfcd+Qbfce-Qbfde+Qcfde)
        +ad*bc*ef*(-Qadbc+Qaebc+Qafbc+Qadbe+Qadbf+Qabef+Qadce+Qadcf+Qacef-Qadef+Qbcde+Qbcdf-Qbcef+Qbdef+Qcdef)
        +ad*be*cf*(+Qadbc+Qacbe+Qabcf-Qadbe+Qadbf+Qafbe+Qadce-Qadcf+Qaecf+Qadef+Qbecd+Qbdcf-Qbecf+Qbedf+Qcfde)
        +ad*bf*ce*(+Qadbc+Qabce+Qacbf+Qadbe-Qadbf+Qaebf-Qadce+Qadcf+Qafce+Qadef+Qbdce+Qbfcd-Qbfce+Qbfde+Qcedf)
        +ae*bc*df*(+Qadbc-Qaebc+Qafbc+Qaebd+Qabdf+Qaebf+Qaecd+Qacdf+Qaecf-Qaedf+Qbcde-Qbcdf+Qbcef+Qbedf+Qcedf)
        +ae*bd*cf*(+Qacbd+Qaebc+Qabcf-Qaebd+Qafbd+Qaebf+Qaecd+Qadcf-Qaecf+Qaedf+Qbdce-Qbdcf+Qbecf+Qbdef+Qcfde)
        +ae*bf*cd*(+Qabcd+Qaebc+Qacbf+Qaebd+Qadbf-Qaebf-Qaecd+Qafcd+Qaecf+Qaedf+Qbecd-Qbfcd+Qbfce+Qbfde+Qcdef)
        +af*bc*de*(+Qadbc+Qaebc-Qafbc+Qabde+Qafbd+Qafbe+Qacde+Qafcd+Qafce-Qafde-Qbcde+Qbcdf+Qbcef+Qbfde+Qcfde)
        +af*bd*ce*(+Qacbd+Qabce+Qafbc+Qaebd-Qafbd+Qafbe+Qadce+Qafcd-Qafce+Qafde-Qbdce+Qbdcf+Qbfce+Qbdef+Qcedf)
        +af*be*cd*(+Qabcd+Qacbe+Qafbc+Qadbe+Qafbd-Qafbe+Qaecd-Qafcd+Qafce+Qafde-Qbecd+Qbfcd+Qbecf+Qbedf+Qcdef)

        -(ab*ac*bc*(de*(df+ef-de)+df*(de+ef-df)+ef*(de+df-ef))+
          ab*ad*bd*(ce*(cf+ef-ce)+cf*(ce+ef-cf)+ef*(ce+cf-ef))+
          ab*ae*be*(cd*(cf+df-cd)+cf*(cd+df-cf)+df*(cd+cf-df))+
          ab*af*bf*(cd*(ce+de-cd)+ce*(cd+de-ce)+de*(cd+ce-de))+
          ac*ad*cd*(be*(bf+ef-be)+bf*(be+ef-bf)+ef*(be+bf-ef))+
          ac*ae*ce*(bd*(bf+df-bd)+bf*(bd+df-bf)+df*(bd+bf-df))+
          ac*af*cf*(bd*(be+de-bd)+be*(bd+de-be)+de*(bd+be-de))+
          ad*ae*de*(bc*(bf+cf-bc)+bf*(bc+cf-bf)+cf*(bc+bf-cf))+
          ad*af*df*(bc*(be+ce-bc)+be*(bc+ce-be)+ce*(bc+be-ce))+
          ae*af*ef*(bc*(bd+cd-bc)+bd*(bc+cd-bd)+cd*(bc+bd-cd))+
          bc*bd*cd*(ae*(af+ef-ae)+af*(ae+ef-af)+ef*(ae+af-ef))+
          bc*be*ce*(ad*(af+df-ad)+af*(ad+df-af)+df*(ad+af-df))+
          bc*bf*cf*(ad*(ae+de-ad)+ae*(ad+de-ae)+de*(ad+ae-de))+
          bd*be*de*(ac*(af+cf-ac)+af*(ac+cf-af)+cf*(ac+af-cf))+
          bd*bf*df*(ac*(ae+ce-ac)+ae*(ac+ce-ae)+ce*(ac+ae-ce))+
          be*bf*ef*(ac*(ad+cd-ac)+ad*(ac+cd-ad)+cd*(ac+ad-cd))+
          cd*ce*de*(ab*(af+bf-ab)+af*(ab+bf-af)+bf*(ab+af-bf))+
          cd*cf*df*(ab*(ae+be-ab)+ae*(ab+be-ae)+be*(ab+ae-be))+
          ce*cf*ef*(ab*(ad+bd-ab)+ad*(ab+bd-ad)+bd*(ab+ad-bd))+
          de*df*ef*(ab*(ac+bc-ab)+ac*(ab+bc-ac)+bc*(ab+ac-bc)))

        -(ab*(acde*(bdce+becd)+acdf*(bdcf+bfcd)+acef*(becf+bfce)+
              adce*(bcde+becd)+adcf*(bcdf+bfcd)+adef*(bedf+bfde)+
              aecd*(bcde+bdce)+aecf*(bcef+bfce)+aedf*(bdef+bfde)+
              afcd*(bcdf+bdcf)+afce*(bcef+becf)+afde*(bdef+bedf))+
          ac*(adbe*(bcde+bdce)+adbf*(bcdf+bdcf)+adef*(cedf+cfde)+
              aebd*(bcde+becd)+aebf*(bcef+becf)+aedf*(cdef+cfde)+
              afbd*(bcdf+bfcd)+afbe*(bcef+bfce)+afde*(cdef+cedf))+
          ad*(aebc*(bdce+becd)+aebf*(bdef+bedf)+aecf*(cdef+cedf)+
              afbc*(bdcf+bfcd)+afbe*(bdef+bfde)+afce*(cdef+cfde))+
          ae*(afbc*(becf+bfce)+afbd*(bedf+bfde)+afcd*(cedf+cfde))+
          bc*(bdef*(cedf+cfde)+bedf*(cdef+cfde)+bfde*(cdef+cedf))+
          bd*(becf*(cdef+cedf)+bfce*(cdef+cfde))+
          be*(bfcd*(cedf+cfde)));
    }

DEVICE inline OverlapReal vok6(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,
                OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,
                OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,
                OverlapReal de,OverlapReal df,OverlapReal dg,
                OverlapReal ef,OverlapReal eg,
                OverlapReal fg)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        OverlapReal abce = ab*ce,acbe = ac*be,aebc = ae*bc;
        OverlapReal abcf = ab*cf,acbf = ac*bf,afbc = af*bc;
        OverlapReal abcg = ab*cg,acbg = ac*bg,agbc = ag*bc;
        OverlapReal abde = ab*de,adbe = ad*be,aebd = ae*bd;
        OverlapReal abdf = ab*df,adbf = ad*bf,afbd = af*bd;
        OverlapReal abdg = ab*dg,adbg = ad*bg,agbd = ag*bd;
        OverlapReal abef = ab*ef,aebf = ae*bf,afbe = af*be;
        OverlapReal abeg = ab*eg,aebg = ae*bg,agbe = ag*be;
        OverlapReal abfg = ab*fg,afbg = af*bg,agbf = ag*bf;
        OverlapReal acde = ac*de,adce = ad*ce,aecd = ae*cd;
        OverlapReal acdf = ac*df,adcf = ad*cf,afcd = af*cd;
        OverlapReal acdg = ac*dg,adcg = ad*cg,agcd = ag*cd;
        OverlapReal acef = ac*ef,aecf = ae*cf,afce = af*ce;
        OverlapReal aceg = ac*eg,aecg = ae*cg,agce = ag*ce;
        OverlapReal acfg = ac*fg,afcg = af*cg,agcf = ag*cf;
        OverlapReal adef = ad*ef,aedf = ae*df,afde = af*de;
        OverlapReal adeg = ad*eg,aedg = ae*dg,agde = ag*de;
        OverlapReal adfg = ad*fg,afdg = af*dg,agdf = ag*df;
        OverlapReal aefg = ae*fg,afeg = af*eg,agef = ag*ef;
        OverlapReal bcde = bc*de,bdce = bd*ce,becd = be*cd;
        OverlapReal bcdf = bc*df,bdcf = bd*cf,bfcd = bf*cd;
        OverlapReal bcdg = bc*dg,bdcg = bd*cg,bgcd = bg*cd;
        OverlapReal bcef = bc*ef,becf = be*cf,bfce = bf*ce;
        OverlapReal bceg = bc*eg,becg = be*cg,bgce = bg*ce;
        OverlapReal bcfg = bc*fg,bfcg = bf*cg,bgcf = bg*cf;
        OverlapReal bdef = bd*ef,bedf = be*df,bfde = bf*de;
        OverlapReal bdeg = bd*eg,bedg = be*dg,bgde = bg*de;
        OverlapReal bdfg = bd*fg,bfdg = bf*dg,bgdf = bg*df;
        OverlapReal befg = be*fg,bfeg = bf*eg,bgef = bg*ef;
        OverlapReal cdef = cd*ef,cedf = ce*df,cfde = cf*de;
        OverlapReal cdeg = cd*eg,cedg = ce*dg,cgde = cg*de;
        OverlapReal cdfg = cd*fg,cfdg = cf*dg,cgdf = cg*df;
        OverlapReal cefg = ce*fg,cfeg = cf*eg,cgef = cg*ef;
        OverlapReal defg = de*fg,dfeg = df*eg,dgef = dg*ef;

        OverlapReal abcdef = ab*cdef,abcedf = ab*cedf,abcfde = ab*cfde;
        OverlapReal acbdef = ac*bdef,acbedf = ac*bedf,acbfde = ac*bfde;
        OverlapReal adbcef = ad*bcef,adbecf = ad*becf,adbfce = ad*bfce;
        OverlapReal aebcdf = ae*bcdf,aebdcf = ae*bdcf,aebfcd = ae*bfcd;
        OverlapReal afbcde = af*bcde,afbdce = af*bdce,afbecd = af*becd;
        OverlapReal abcdeg = ab*cdeg,abcedg = ab*cedg,abcgde = ab*cgde;
        OverlapReal acbdeg = ac*bdeg,acbedg = ac*bedg,acbgde = ac*bgde;
        OverlapReal adbceg = ad*bceg,adbecg = ad*becg,adbgce = ad*bgce;
        OverlapReal aebcdg = ae*bcdg,aebdcg = ae*bdcg,aebgcd = ae*bgcd;
        OverlapReal agbcde = ag*bcde,agbdce = ag*bdce,agbecd = ag*becd;
        OverlapReal abcdfg = ab*cdfg,abcfdg = ab*cfdg,abcgdf = ab*cgdf;
        OverlapReal acbdfg = ac*bdfg,acbfdg = ac*bfdg,acbgdf = ac*bgdf;
        OverlapReal adbcfg = ad*bcfg,adbfcg = ad*bfcg,adbgcf = ad*bgcf;
        OverlapReal afbcdg = af*bcdg,afbdcg = af*bdcg,afbgcd = af*bgcd;
        OverlapReal agbcdf = ag*bcdf,agbdcf = ag*bdcf,agbfcd = ag*bfcd;
        OverlapReal abcefg = ab*cefg,abcfeg = ab*cfeg,abcgef = ab*cgef;
        OverlapReal acbefg = ac*befg,acbfeg = ac*bfeg,acbgef = ac*bgef;
        OverlapReal aebcfg = ae*bcfg,aebfcg = ae*bfcg,aebgcf = ae*bgcf;
        OverlapReal afbceg = af*bceg,afbecg = af*becg,afbgce = af*bgce;
        OverlapReal agbcef = ag*bcef,agbecf = ag*becf,agbfce = ag*bfce;
        OverlapReal abdefg = ab*defg,abdfeg = ab*dfeg,abdgef = ab*dgef;
        OverlapReal adbefg = ad*befg,adbfeg = ad*bfeg,adbgef = ad*bgef;
        OverlapReal aebdfg = ae*bdfg,aebfdg = ae*bfdg,aebgdf = ae*bgdf;
        OverlapReal afbdeg = af*bdeg,afbedg = af*bedg,afbgde = af*bgde;
        OverlapReal agbdef = ag*bdef,agbedf = ag*bedf,agbfde = ag*bfde;
        OverlapReal acdefg = ac*defg,acdfeg = ac*dfeg,acdgef = ac*dgef;
        OverlapReal adcefg = ad*cefg,adcfeg = ad*cfeg,adcgef = ad*cgef;
        OverlapReal aecdfg = ae*cdfg,aecfdg = ae*cfdg,aecgdf = ae*cgdf;
        OverlapReal afcdeg = af*cdeg,afcedg = af*cedg,afcgde = af*cgde;
        OverlapReal agcdef = ag*cdef,agcedf = ag*cedf,agcfde = ag*cfde;
        OverlapReal bcdefg = bc*defg,bcdfeg = bc*dfeg,bcdgef = bc*dgef;
        OverlapReal bdcefg = bd*cefg,bdcfeg = bd*cfeg,bdcgef = bd*cgef;
        OverlapReal becdfg = be*cdfg,becfdg = be*cfdg,becgdf = be*cgdf;
        OverlapReal bfcdeg = bf*cdeg,bfcedg = bf*cedg,bfcgde = bf*cgde;
        OverlapReal bgcdef = bg*cdef,bgcedf = bg*cedf,bgcfde = bg*cfde;

        OverlapReal Qabcdef = abcdef+acbedf+acbfde+adbecf+adbfce+aebcdf+aebdcf+afbcde+afbdce-abcedf-abcfde-acbdef-adbcef-aebfcd-afbecd;
        OverlapReal Qabcedf = abcedf+acbdef+acbfde+adbcef+adbecf+aebdcf+aebfcd+afbcde+afbecd-abcdef-abcfde-acbedf-adbfce-aebcdf-afbdce;
        OverlapReal Qabcfde = abcfde+acbdef+acbedf+adbcef+adbfce+aebcdf+aebfcd+afbdce+afbecd-abcdef-abcedf-acbfde-adbecf-aebdcf-afbcde;
        OverlapReal Qacbdef = abcedf+abcfde+acbdef+adbecf+adbfce+aebcdf+aebfcd+afbcde+afbecd-abcdef-acbedf-acbfde-adbcef-aebdcf-afbdce;
        OverlapReal Qacbedf = abcdef+abcfde+acbedf+adbcef+adbfce+aebdcf+aebfcd+afbcde+afbdce-abcedf-acbdef-acbfde-adbecf-aebcdf-afbecd;
        OverlapReal Qacbfde = abcdef+abcedf+acbfde+adbcef+adbecf+aebcdf+aebdcf+afbdce+afbecd-abcfde-acbdef-acbedf-adbfce-aebfcd-afbcde;
        OverlapReal Qadbcef = abcedf+abcfde+acbedf+acbfde+adbcef+aebdcf+aebfcd+afbdce+afbecd-abcdef-acbdef-adbecf-adbfce-aebcdf-afbcde;
        OverlapReal Qadbecf = abcdef+abcedf+acbdef+acbfde+adbecf+aebcdf+aebfcd+afbcde+afbdce-abcfde-acbedf-adbcef-adbfce-aebdcf-afbecd;
        OverlapReal Qadbfce = abcdef+abcfde+acbdef+acbedf+adbfce+aebcdf+aebdcf+afbcde+afbecd-abcedf-acbfde-adbcef-adbecf-aebfcd-afbdce;
        OverlapReal Qaebcdf = abcdef+abcfde+acbdef+acbfde+adbecf+adbfce+aebcdf+afbdce+afbecd-abcedf-acbedf-adbcef-aebdcf-aebfcd-afbcde;
        OverlapReal Qaebdcf = abcdef+abcedf+acbedf+acbfde+adbcef+adbfce+aebdcf+afbcde+afbecd-abcfde-acbdef-adbecf-aebcdf-aebfcd-afbdce;
        OverlapReal Qaebfcd = abcedf+abcfde+acbdef+acbedf+adbcef+adbecf+aebfcd+afbcde+afbdce-abcdef-acbfde-adbfce-aebcdf-aebdcf-afbecd;
        OverlapReal Qafbcde = abcdef+abcedf+acbdef+acbedf+adbecf+adbfce+aebdcf+aebfcd+afbcde-abcfde-acbfde-adbcef-aebcdf-afbdce-afbecd;
        OverlapReal Qafbdce = abcdef+abcfde+acbedf+acbfde+adbcef+adbecf+aebcdf+aebfcd+afbdce-abcedf-acbdef-adbfce-aebdcf-afbcde-afbecd;
        OverlapReal Qafbecd = abcedf+abcfde+acbdef+acbfde+adbcef+adbfce+aebcdf+aebdcf+afbecd-abcdef-acbedf-adbecf-aebfcd-afbcde-afbdce;
        OverlapReal Qabcdeg = abcdeg+acbedg+acbgde+adbecg+adbgce+aebcdg+aebdcg+agbcde+agbdce-abcedg-abcgde-acbdeg-adbceg-aebgcd-agbecd;
        OverlapReal Qabcedg = abcedg+acbdeg+acbgde+adbceg+adbecg+aebdcg+aebgcd+agbcde+agbecd-abcdeg-abcgde-acbedg-adbgce-aebcdg-agbdce;
        OverlapReal Qabcgde = abcgde+acbdeg+acbedg+adbceg+adbgce+aebcdg+aebgcd+agbdce+agbecd-abcdeg-abcedg-acbgde-adbecg-aebdcg-agbcde;
        OverlapReal Qacbdeg = abcedg+abcgde+acbdeg+adbecg+adbgce+aebcdg+aebgcd+agbcde+agbecd-abcdeg-acbedg-acbgde-adbceg-aebdcg-agbdce;
        OverlapReal Qacbedg = abcdeg+abcgde+acbedg+adbceg+adbgce+aebdcg+aebgcd+agbcde+agbdce-abcedg-acbdeg-acbgde-adbecg-aebcdg-agbecd;
        OverlapReal Qacbgde = abcdeg+abcedg+acbgde+adbceg+adbecg+aebcdg+aebdcg+agbdce+agbecd-abcgde-acbdeg-acbedg-adbgce-aebgcd-agbcde;
        OverlapReal Qadbceg = abcedg+abcgde+acbedg+acbgde+adbceg+aebdcg+aebgcd+agbdce+agbecd-abcdeg-acbdeg-adbecg-adbgce-aebcdg-agbcde;
        OverlapReal Qadbecg = abcdeg+abcedg+acbdeg+acbgde+adbecg+aebcdg+aebgcd+agbcde+agbdce-abcgde-acbedg-adbceg-adbgce-aebdcg-agbecd;
        OverlapReal Qadbgce = abcdeg+abcgde+acbdeg+acbedg+adbgce+aebcdg+aebdcg+agbcde+agbecd-abcedg-acbgde-adbceg-adbecg-aebgcd-agbdce;
        OverlapReal Qaebcdg = abcdeg+abcgde+acbdeg+acbgde+adbecg+adbgce+aebcdg+agbdce+agbecd-abcedg-acbedg-adbceg-aebdcg-aebgcd-agbcde;
        OverlapReal Qaebdcg = abcdeg+abcedg+acbedg+acbgde+adbceg+adbgce+aebdcg+agbcde+agbecd-abcgde-acbdeg-adbecg-aebcdg-aebgcd-agbdce;
        OverlapReal Qaebgcd = abcedg+abcgde+acbdeg+acbedg+adbceg+adbecg+aebgcd+agbcde+agbdce-abcdeg-acbgde-adbgce-aebcdg-aebdcg-agbecd;
        OverlapReal Qagbcde = abcdeg+abcedg+acbdeg+acbedg+adbecg+adbgce+aebdcg+aebgcd+agbcde-abcgde-acbgde-adbceg-aebcdg-agbdce-agbecd;
        OverlapReal Qagbdce = abcdeg+abcgde+acbedg+acbgde+adbceg+adbecg+aebcdg+aebgcd+agbdce-abcedg-acbdeg-adbgce-aebdcg-agbcde-agbecd;
        OverlapReal Qagbecd = abcedg+abcgde+acbdeg+acbgde+adbceg+adbgce+aebcdg+aebdcg+agbecd-abcdeg-acbedg-adbecg-aebgcd-agbcde-agbdce;
        OverlapReal Qabcdfg = abcdfg+acbfdg+acbgdf+adbfcg+adbgcf+afbcdg+afbdcg+agbcdf+agbdcf-abcfdg-abcgdf-acbdfg-adbcfg-afbgcd-agbfcd;
        OverlapReal Qabcfdg = abcfdg+acbdfg+acbgdf+adbcfg+adbfcg+afbdcg+afbgcd+agbcdf+agbfcd-abcdfg-abcgdf-acbfdg-adbgcf-afbcdg-agbdcf;
        OverlapReal Qabcgdf = abcgdf+acbdfg+acbfdg+adbcfg+adbgcf+afbcdg+afbgcd+agbdcf+agbfcd-abcdfg-abcfdg-acbgdf-adbfcg-afbdcg-agbcdf;
        OverlapReal Qacbdfg = abcfdg+abcgdf+acbdfg+adbfcg+adbgcf+afbcdg+afbgcd+agbcdf+agbfcd-abcdfg-acbfdg-acbgdf-adbcfg-afbdcg-agbdcf;
        OverlapReal Qacbfdg = abcdfg+abcgdf+acbfdg+adbcfg+adbgcf+afbdcg+afbgcd+agbcdf+agbdcf-abcfdg-acbdfg-acbgdf-adbfcg-afbcdg-agbfcd;
        OverlapReal Qacbgdf = abcdfg+abcfdg+acbgdf+adbcfg+adbfcg+afbcdg+afbdcg+agbdcf+agbfcd-abcgdf-acbdfg-acbfdg-adbgcf-afbgcd-agbcdf;
        OverlapReal Qadbcfg = abcfdg+abcgdf+acbfdg+acbgdf+adbcfg+afbdcg+afbgcd+agbdcf+agbfcd-abcdfg-acbdfg-adbfcg-adbgcf-afbcdg-agbcdf;
        OverlapReal Qadbfcg = abcdfg+abcfdg+acbdfg+acbgdf+adbfcg+afbcdg+afbgcd+agbcdf+agbdcf-abcgdf-acbfdg-adbcfg-adbgcf-afbdcg-agbfcd;
        OverlapReal Qadbgcf = abcdfg+abcgdf+acbdfg+acbfdg+adbgcf+afbcdg+afbdcg+agbcdf+agbfcd-abcfdg-acbgdf-adbcfg-adbfcg-afbgcd-agbdcf;
        OverlapReal Qafbcdg = abcdfg+abcgdf+acbdfg+acbgdf+adbfcg+adbgcf+afbcdg+agbdcf+agbfcd-abcfdg-acbfdg-adbcfg-afbdcg-afbgcd-agbcdf;
        OverlapReal Qafbdcg = abcdfg+abcfdg+acbfdg+acbgdf+adbcfg+adbgcf+afbdcg+agbcdf+agbfcd-abcgdf-acbdfg-adbfcg-afbcdg-afbgcd-agbdcf;
        OverlapReal Qafbgcd = abcfdg+abcgdf+acbdfg+acbfdg+adbcfg+adbfcg+afbgcd+agbcdf+agbdcf-abcdfg-acbgdf-adbgcf-afbcdg-afbdcg-agbfcd;
        OverlapReal Qagbcdf = abcdfg+abcfdg+acbdfg+acbfdg+adbfcg+adbgcf+afbdcg+afbgcd+agbcdf-abcgdf-acbgdf-adbcfg-afbcdg-agbdcf-agbfcd;
        OverlapReal Qagbdcf = abcdfg+abcgdf+acbfdg+acbgdf+adbcfg+adbfcg+afbcdg+afbgcd+agbdcf-abcfdg-acbdfg-adbgcf-afbdcg-agbcdf-agbfcd;
        OverlapReal Qagbfcd = abcfdg+abcgdf+acbdfg+acbgdf+adbcfg+adbgcf+afbcdg+afbdcg+agbfcd-abcdfg-acbfdg-adbfcg-afbgcd-agbcdf-agbdcf;
        OverlapReal Qabcefg = abcefg+acbfeg+acbgef+aebfcg+aebgcf+afbceg+afbecg+agbcef+agbecf-abcfeg-abcgef-acbefg-aebcfg-afbgce-agbfce;
        OverlapReal Qabcfeg = abcfeg+acbefg+acbgef+aebcfg+aebfcg+afbecg+afbgce+agbcef+agbfce-abcefg-abcgef-acbfeg-aebgcf-afbceg-agbecf;
        OverlapReal Qabcgef = abcgef+acbefg+acbfeg+aebcfg+aebgcf+afbceg+afbgce+agbecf+agbfce-abcefg-abcfeg-acbgef-aebfcg-afbecg-agbcef;
        OverlapReal Qacbefg = abcfeg+abcgef+acbefg+aebfcg+aebgcf+afbceg+afbgce+agbcef+agbfce-abcefg-acbfeg-acbgef-aebcfg-afbecg-agbecf;
        OverlapReal Qacbfeg = abcefg+abcgef+acbfeg+aebcfg+aebgcf+afbecg+afbgce+agbcef+agbecf-abcfeg-acbefg-acbgef-aebfcg-afbceg-agbfce;
        OverlapReal Qacbgef = abcefg+abcfeg+acbgef+aebcfg+aebfcg+afbceg+afbecg+agbecf+agbfce-abcgef-acbefg-acbfeg-aebgcf-afbgce-agbcef;
        OverlapReal Qaebcfg = abcfeg+abcgef+acbfeg+acbgef+aebcfg+afbecg+afbgce+agbecf+agbfce-abcefg-acbefg-aebfcg-aebgcf-afbceg-agbcef;
        OverlapReal Qaebfcg = abcefg+abcfeg+acbefg+acbgef+aebfcg+afbceg+afbgce+agbcef+agbecf-abcgef-acbfeg-aebcfg-aebgcf-afbecg-agbfce;
        OverlapReal Qaebgcf = abcefg+abcgef+acbefg+acbfeg+aebgcf+afbceg+afbecg+agbcef+agbfce-abcfeg-acbgef-aebcfg-aebfcg-afbgce-agbecf;
        OverlapReal Qafbceg = abcefg+abcgef+acbefg+acbgef+aebfcg+aebgcf+afbceg+agbecf+agbfce-abcfeg-acbfeg-aebcfg-afbecg-afbgce-agbcef;
        OverlapReal Qafbecg = abcefg+abcfeg+acbfeg+acbgef+aebcfg+aebgcf+afbecg+agbcef+agbfce-abcgef-acbefg-aebfcg-afbceg-afbgce-agbecf;
        OverlapReal Qafbgce = abcfeg+abcgef+acbefg+acbfeg+aebcfg+aebfcg+afbgce+agbcef+agbecf-abcefg-acbgef-aebgcf-afbceg-afbecg-agbfce;
        OverlapReal Qagbcef = abcefg+abcfeg+acbefg+acbfeg+aebfcg+aebgcf+afbecg+afbgce+agbcef-abcgef-acbgef-aebcfg-afbceg-agbecf-agbfce;
        OverlapReal Qagbecf = abcefg+abcgef+acbfeg+acbgef+aebcfg+aebfcg+afbceg+afbgce+agbecf-abcfeg-acbefg-aebgcf-afbecg-agbcef-agbfce;
        OverlapReal Qagbfce = abcfeg+abcgef+acbefg+acbgef+aebcfg+aebgcf+afbceg+afbecg+agbfce-abcefg-acbfeg-aebfcg-afbgce-agbcef-agbecf;
        OverlapReal Qabdefg = abdefg+adbfeg+adbgef+aebfdg+aebgdf+afbdeg+afbedg+agbdef+agbedf-abdfeg-abdgef-adbefg-aebdfg-afbgde-agbfde;
        OverlapReal Qabdfeg = abdfeg+adbefg+adbgef+aebdfg+aebfdg+afbedg+afbgde+agbdef+agbfde-abdefg-abdgef-adbfeg-aebgdf-afbdeg-agbedf;
        OverlapReal Qabdgef = abdgef+adbefg+adbfeg+aebdfg+aebgdf+afbdeg+afbgde+agbedf+agbfde-abdefg-abdfeg-adbgef-aebfdg-afbedg-agbdef;
        OverlapReal Qadbefg = abdfeg+abdgef+adbefg+aebfdg+aebgdf+afbdeg+afbgde+agbdef+agbfde-abdefg-adbfeg-adbgef-aebdfg-afbedg-agbedf;
        OverlapReal Qadbfeg = abdefg+abdgef+adbfeg+aebdfg+aebgdf+afbedg+afbgde+agbdef+agbedf-abdfeg-adbefg-adbgef-aebfdg-afbdeg-agbfde;
        OverlapReal Qadbgef = abdefg+abdfeg+adbgef+aebdfg+aebfdg+afbdeg+afbedg+agbedf+agbfde-abdgef-adbefg-adbfeg-aebgdf-afbgde-agbdef;
        OverlapReal Qaebdfg = abdfeg+abdgef+adbfeg+adbgef+aebdfg+afbedg+afbgde+agbedf+agbfde-abdefg-adbefg-aebfdg-aebgdf-afbdeg-agbdef;
        OverlapReal Qaebfdg = abdefg+abdfeg+adbefg+adbgef+aebfdg+afbdeg+afbgde+agbdef+agbedf-abdgef-adbfeg-aebdfg-aebgdf-afbedg-agbfde;
        OverlapReal Qaebgdf = abdefg+abdgef+adbefg+adbfeg+aebgdf+afbdeg+afbedg+agbdef+agbfde-abdfeg-adbgef-aebdfg-aebfdg-afbgde-agbedf;
        OverlapReal Qafbdeg = abdefg+abdgef+adbefg+adbgef+aebfdg+aebgdf+afbdeg+agbedf+agbfde-abdfeg-adbfeg-aebdfg-afbedg-afbgde-agbdef;
        OverlapReal Qafbedg = abdefg+abdfeg+adbfeg+adbgef+aebdfg+aebgdf+afbedg+agbdef+agbfde-abdgef-adbefg-aebfdg-afbdeg-afbgde-agbedf;
        OverlapReal Qafbgde = abdfeg+abdgef+adbefg+adbfeg+aebdfg+aebfdg+afbgde+agbdef+agbedf-abdefg-adbgef-aebgdf-afbdeg-afbedg-agbfde;
        OverlapReal Qagbdef = abdefg+abdfeg+adbefg+adbfeg+aebfdg+aebgdf+afbedg+afbgde+agbdef-abdgef-adbgef-aebdfg-afbdeg-agbedf-agbfde;
        OverlapReal Qagbedf = abdefg+abdgef+adbfeg+adbgef+aebdfg+aebfdg+afbdeg+afbgde+agbedf-abdfeg-adbefg-aebgdf-afbedg-agbdef-agbfde;
        OverlapReal Qagbfde = abdfeg+abdgef+adbefg+adbgef+aebdfg+aebgdf+afbdeg+afbedg+agbfde-abdefg-adbfeg-aebfdg-afbgde-agbdef-agbedf;
        OverlapReal Qacdefg = acdefg+adcfeg+adcgef+aecfdg+aecgdf+afcdeg+afcedg+agcdef+agcedf-acdfeg-acdgef-adcefg-aecdfg-afcgde-agcfde;
        OverlapReal Qacdfeg = acdfeg+adcefg+adcgef+aecdfg+aecfdg+afcedg+afcgde+agcdef+agcfde-acdefg-acdgef-adcfeg-aecgdf-afcdeg-agcedf;
        OverlapReal Qacdgef = acdgef+adcefg+adcfeg+aecdfg+aecgdf+afcdeg+afcgde+agcedf+agcfde-acdefg-acdfeg-adcgef-aecfdg-afcedg-agcdef;
        OverlapReal Qadcefg = acdfeg+acdgef+adcefg+aecfdg+aecgdf+afcdeg+afcgde+agcdef+agcfde-acdefg-adcfeg-adcgef-aecdfg-afcedg-agcedf;
        OverlapReal Qadcfeg = acdefg+acdgef+adcfeg+aecdfg+aecgdf+afcedg+afcgde+agcdef+agcedf-acdfeg-adcefg-adcgef-aecfdg-afcdeg-agcfde;
        OverlapReal Qadcgef = acdefg+acdfeg+adcgef+aecdfg+aecfdg+afcdeg+afcedg+agcedf+agcfde-acdgef-adcefg-adcfeg-aecgdf-afcgde-agcdef;
        OverlapReal Qaecdfg = acdfeg+acdgef+adcfeg+adcgef+aecdfg+afcedg+afcgde+agcedf+agcfde-acdefg-adcefg-aecfdg-aecgdf-afcdeg-agcdef;
        OverlapReal Qaecfdg = acdefg+acdfeg+adcefg+adcgef+aecfdg+afcdeg+afcgde+agcdef+agcedf-acdgef-adcfeg-aecdfg-aecgdf-afcedg-agcfde;
        OverlapReal Qaecgdf = acdefg+acdgef+adcefg+adcfeg+aecgdf+afcdeg+afcedg+agcdef+agcfde-acdfeg-adcgef-aecdfg-aecfdg-afcgde-agcedf;
        OverlapReal Qafcdeg = acdefg+acdgef+adcefg+adcgef+aecfdg+aecgdf+afcdeg+agcedf+agcfde-acdfeg-adcfeg-aecdfg-afcedg-afcgde-agcdef;
        OverlapReal Qafcedg = acdefg+acdfeg+adcfeg+adcgef+aecdfg+aecgdf+afcedg+agcdef+agcfde-acdgef-adcefg-aecfdg-afcdeg-afcgde-agcedf;
        OverlapReal Qafcgde = acdfeg+acdgef+adcefg+adcfeg+aecdfg+aecfdg+afcgde+agcdef+agcedf-acdefg-adcgef-aecgdf-afcdeg-afcedg-agcfde;
        OverlapReal Qagcdef = acdefg+acdfeg+adcefg+adcfeg+aecfdg+aecgdf+afcedg+afcgde+agcdef-acdgef-adcgef-aecdfg-afcdeg-agcedf-agcfde;
        OverlapReal Qagcedf = acdefg+acdgef+adcfeg+adcgef+aecdfg+aecfdg+afcdeg+afcgde+agcedf-acdfeg-adcefg-aecgdf-afcedg-agcdef-agcfde;
        OverlapReal Qagcfde = acdfeg+acdgef+adcefg+adcgef+aecdfg+aecgdf+afcdeg+afcedg+agcfde-acdefg-adcfeg-aecfdg-afcgde-agcdef-agcedf;
        OverlapReal Qbcdefg = bcdefg+bdcfeg+bdcgef+becfdg+becgdf+bfcdeg+bfcedg+bgcdef+bgcedf-bcdfeg-bcdgef-bdcefg-becdfg-bfcgde-bgcfde;
        OverlapReal Qbcdfeg = bcdfeg+bdcefg+bdcgef+becdfg+becfdg+bfcedg+bfcgde+bgcdef+bgcfde-bcdefg-bcdgef-bdcfeg-becgdf-bfcdeg-bgcedf;
        OverlapReal Qbcdgef = bcdgef+bdcefg+bdcfeg+becdfg+becgdf+bfcdeg+bfcgde+bgcedf+bgcfde-bcdefg-bcdfeg-bdcgef-becfdg-bfcedg-bgcdef;
        OverlapReal Qbdcefg = bcdfeg+bcdgef+bdcefg+becfdg+becgdf+bfcdeg+bfcgde+bgcdef+bgcfde-bcdefg-bdcfeg-bdcgef-becdfg-bfcedg-bgcedf;
        OverlapReal Qbdcfeg = bcdefg+bcdgef+bdcfeg+becdfg+becgdf+bfcedg+bfcgde+bgcdef+bgcedf-bcdfeg-bdcefg-bdcgef-becfdg-bfcdeg-bgcfde;
        OverlapReal Qbdcgef = bcdefg+bcdfeg+bdcgef+becdfg+becfdg+bfcdeg+bfcedg+bgcedf+bgcfde-bcdgef-bdcefg-bdcfeg-becgdf-bfcgde-bgcdef;
        OverlapReal Qbecdfg = bcdfeg+bcdgef+bdcfeg+bdcgef+becdfg+bfcedg+bfcgde+bgcedf+bgcfde-bcdefg-bdcefg-becfdg-becgdf-bfcdeg-bgcdef;
        OverlapReal Qbecfdg = bcdefg+bcdfeg+bdcefg+bdcgef+becfdg+bfcdeg+bfcgde+bgcdef+bgcedf-bcdgef-bdcfeg-becdfg-becgdf-bfcedg-bgcfde;
        OverlapReal Qbecgdf = bcdefg+bcdgef+bdcefg+bdcfeg+becgdf+bfcdeg+bfcedg+bgcdef+bgcfde-bcdfeg-bdcgef-becdfg-becfdg-bfcgde-bgcedf;
        OverlapReal Qbfcdeg = bcdefg+bcdgef+bdcefg+bdcgef+becfdg+becgdf+bfcdeg+bgcedf+bgcfde-bcdfeg-bdcfeg-becdfg-bfcedg-bfcgde-bgcdef;
        OverlapReal Qbfcedg = bcdefg+bcdfeg+bdcfeg+bdcgef+becdfg+becgdf+bfcedg+bgcdef+bgcfde-bcdgef-bdcefg-becfdg-bfcdeg-bfcgde-bgcedf;
        OverlapReal Qbfcgde = bcdfeg+bcdgef+bdcefg+bdcfeg+becdfg+becfdg+bfcgde+bgcdef+bgcedf-bcdefg-bdcgef-becgdf-bfcdeg-bfcedg-bgcfde;
        OverlapReal Qbgcdef = bcdefg+bcdfeg+bdcefg+bdcfeg+becfdg+becgdf+bfcedg+bfcgde+bgcdef-bcdgef-bdcgef-becdfg-bfcdeg-bgcedf-bgcfde;
        OverlapReal Qbgcedf = bcdefg+bcdgef+bdcfeg+bdcgef+becdfg+becfdg+bfcdeg+bfcgde+bgcedf-bcdfeg-bdcefg-becgdf-bfcedg-bgcdef-bgcfde;
        OverlapReal Qbgcfde = bcdfeg+bcdgef+bdcefg+bdcgef+becdfg+becgdf+bfcdeg+bfcedg+bgcfde-bcdefg-bdcfeg-becfdg-bfcgde-bgcdef-bgcedf;

        return(abcdef*(Qabcdeg+Qabcdfg+Qabcgef+Qabdgef+Qagcdef+Qbgcdef-Qabcdef)+
               abcedf*(Qabcedg+Qabcgdf+Qabcefg+Qabdfeg+Qagcedf+Qbgcedf-Qabcedf)+
               abcfde*(Qabcgde+Qabcfdg+Qabcfeg+Qabdefg+Qagcfde+Qbgcfde-Qabcfde)+
               acbdef*(Qacbdeg+Qacbdfg+Qacbgef+Qagbdef+Qacdgef+Qbdcgef-Qacbdef)+
               acbedf*(Qacbedg+Qacbgdf+Qacbefg+Qagbedf+Qacdfeg+Qbecgdf-Qacbedf)+
               acbfde*(Qacbgde+Qacbfdg+Qacbfeg+Qagbfde+Qacdefg+Qbfcgde-Qacbfde)+
               adbcef*(Qadbceg+Qadbcfg+Qagbcef+Qadbgef+Qadcgef+Qbcdgef-Qadbcef)+
               adbecf*(Qadbecg+Qadbgcf+Qagbecf+Qadbefg+Qadcfeg+Qbecfdg-Qadbecf)+
               adbfce*(Qadbgce+Qadbfcg+Qagbfce+Qadbfeg+Qadcefg+Qbfcedg-Qadbfce)+
               aebcdf*(Qaebcdg+Qagbcdf+Qaebcfg+Qaebgdf+Qaecgdf+Qbcdfeg-Qaebcdf)+
               aebdcf*(Qaebdcg+Qagbdcf+Qaebgcf+Qaebdfg+Qaecfdg+Qbdcfeg-Qaebdcf)+
               aebfcd*(Qaebgcd+Qagbfcd+Qaebfcg+Qaebfdg+Qaecdfg+Qbfcdeg-Qaebfcd)+
               afbcde*(Qagbcde+Qafbcdg+Qafbceg+Qafbgde+Qafcgde+Qbcdefg-Qafbcde)+
               afbdce*(Qagbdce+Qafbdcg+Qafbgce+Qafbdeg+Qafcedg+Qbdcefg-Qafbdce)+
               afbecd*(Qagbecd+Qafbgcd+Qafbecg+Qafbedg+Qafcdeg+Qbecdfg-Qafbecd)+
               abcdeg*(Qabcdef+Qabcdfg+Qabcfeg+Qabdfeg+Qafcdeg+Qbfcdeg-Qabcdeg)+
               abcedg*(Qabcedf+Qabcfdg+Qabcefg+Qabdgef+Qafcedg+Qbfcedg-Qabcedg)+
               abcgde*(Qabcfde+Qabcgdf+Qabcgef+Qabdefg+Qafcgde+Qbfcgde-Qabcgde)+
               acbdeg*(Qacbdef+Qacbdfg+Qacbfeg+Qafbdeg+Qacdfeg+Qbdcfeg-Qacbdeg)+
               acbedg*(Qacbedf+Qacbfdg+Qacbefg+Qafbedg+Qacdgef+Qbecfdg-Qacbedg)+
               acbgde*(Qacbfde+Qacbgdf+Qacbgef+Qafbgde+Qacdefg+Qbgcfde-Qacbgde)+
               adbceg*(Qadbcef+Qadbcfg+Qafbceg+Qadbfeg+Qadcfeg+Qbcdfeg-Qadbceg)+
               adbecg*(Qadbecf+Qadbfcg+Qafbecg+Qadbefg+Qadcgef+Qbecgdf-Qadbecg)+
               adbgce*(Qadbfce+Qadbgcf+Qafbgce+Qadbgef+Qadcefg+Qbgcedf-Qadbgce)+
               aebcdg*(Qaebcdf+Qafbcdg+Qaebcfg+Qaebfdg+Qaecfdg+Qbcdgef-Qaebcdg)+
               aebdcg*(Qaebdcf+Qafbdcg+Qaebfcg+Qaebdfg+Qaecgdf+Qbdcgef-Qaebdcg)+
               aebgcd*(Qaebfcd+Qafbgcd+Qaebgcf+Qaebgdf+Qaecdfg+Qbgcdef-Qaebgcd)+
               agbcde*(Qafbcde+Qagbcdf+Qagbcef+Qagbfde+Qagcfde+Qbcdefg-Qagbcde)+
               agbdce*(Qafbdce+Qagbdcf+Qagbfce+Qagbdef+Qagcedf+Qbdcefg-Qagbdce)+
               agbecd*(Qafbecd+Qagbfcd+Qagbecf+Qagbedf+Qagcdef+Qbecdfg-Qagbecd)+
               abcdfg*(Qabcdef+Qabcdeg+Qabcefg+Qabdefg+Qaecdfg+Qbecdfg-Qabcdfg)+
               abcfdg*(Qabcfde+Qabcedg+Qabcfeg+Qabdgef+Qaecfdg+Qbecfdg-Qabcfdg)+
               abcgdf*(Qabcedf+Qabcgde+Qabcgef+Qabdfeg+Qaecgdf+Qbecgdf-Qabcgdf)+
               acbdfg*(Qacbdef+Qacbdeg+Qacbefg+Qaebdfg+Qacdefg+Qbdcefg-Qacbdfg)+
               acbfdg*(Qacbfde+Qacbedg+Qacbfeg+Qaebfdg+Qacdgef+Qbfcedg-Qacbfdg)+
               acbgdf*(Qacbedf+Qacbgde+Qacbgef+Qaebgdf+Qacdfeg+Qbgcedf-Qacbgdf)+
               adbcfg*(Qadbcef+Qadbceg+Qaebcfg+Qadbefg+Qadcefg+Qbcdefg-Qadbcfg)+
               adbfcg*(Qadbfce+Qadbecg+Qaebfcg+Qadbfeg+Qadcgef+Qbfcgde-Qadbfcg)+
               adbgcf*(Qadbecf+Qadbgce+Qaebgcf+Qadbgef+Qadcfeg+Qbgcfde-Qadbgcf)+
               afbcdg*(Qafbcde+Qaebcdg+Qafbceg+Qafbedg+Qafcedg+Qbcdgef-Qafbcdg)+
               afbdcg*(Qafbdce+Qaebdcg+Qafbecg+Qafbdeg+Qafcgde+Qbdcgef-Qafbdcg)+
               afbgcd*(Qafbecd+Qaebgcd+Qafbgce+Qafbgde+Qafcdeg+Qbgcdef-Qafbgcd)+
               agbcdf*(Qaebcdf+Qagbcde+Qagbcef+Qagbedf+Qagcedf+Qbcdfeg-Qagbcdf)+
               agbdcf*(Qaebdcf+Qagbdce+Qagbecf+Qagbdef+Qagcfde+Qbdcfeg-Qagbdcf)+
               agbfcd*(Qaebfcd+Qagbecd+Qagbfce+Qagbfde+Qagcdef+Qbfcdeg-Qagbfcd)+
               abcefg*(Qabcedf+Qabcedg+Qabcdfg+Qabdefg+Qadcefg+Qbdcefg-Qabcefg)+
               abcfeg*(Qabcfde+Qabcdeg+Qabcfdg+Qabdfeg+Qadcfeg+Qbdcfeg-Qabcfeg)+
               abcgef*(Qabcdef+Qabcgde+Qabcgdf+Qabdgef+Qadcgef+Qbdcgef-Qabcgef)+
               acbefg*(Qacbedf+Qacbedg+Qacbdfg+Qadbefg+Qacdefg+Qbecdfg-Qacbefg)+
               acbfeg*(Qacbfde+Qacbdeg+Qacbfdg+Qadbfeg+Qacdfeg+Qbfcdeg-Qacbfeg)+
               acbgef*(Qacbdef+Qacbgde+Qacbgdf+Qadbgef+Qacdgef+Qbgcdef-Qacbgef)+
               aebcfg*(Qaebcdf+Qaebcdg+Qadbcfg+Qaebdfg+Qaecdfg+Qbcdefg-Qaebcfg)+
               aebfcg*(Qaebfcd+Qaebdcg+Qadbfcg+Qaebfdg+Qaecgdf+Qbfcgde-Qaebfcg)+
               aebgcf*(Qaebdcf+Qaebgcd+Qadbgcf+Qaebgdf+Qaecfdg+Qbgcfde-Qaebgcf)+
               afbceg*(Qafbcde+Qadbceg+Qafbcdg+Qafbdeg+Qafcdeg+Qbcdfeg-Qafbceg)+
               afbecg*(Qafbecd+Qadbecg+Qafbdcg+Qafbedg+Qafcgde+Qbecgdf-Qafbecg)+
               afbgce*(Qafbdce+Qadbgce+Qafbgcd+Qafbgde+Qafcedg+Qbgcedf-Qafbgce)+
               agbcef*(Qadbcef+Qagbcde+Qagbcdf+Qagbdef+Qagcdef+Qbcdgef-Qagbcef)+
               agbecf*(Qadbecf+Qagbecd+Qagbdcf+Qagbedf+Qagcfde+Qbecfdg-Qagbecf)+
               agbfce*(Qadbfce+Qagbdce+Qagbfcd+Qagbfde+Qagcedf+Qbfcedg-Qagbfce)+
               abdefg*(Qabcfde+Qabcgde+Qabcdfg+Qabcefg+Qacdefg+Qbcdefg-Qabdefg)+
               abdfeg*(Qabcedf+Qabcdeg+Qabcgdf+Qabcfeg+Qacdfeg+Qbcdfeg-Qabdfeg)+
               abdgef*(Qabcdef+Qabcedg+Qabcfdg+Qabcgef+Qacdgef+Qbcdgef-Qabdgef)+
               adbefg*(Qadbecf+Qadbecg+Qadbcfg+Qacbefg+Qadcefg+Qbecdfg-Qadbefg)+
               adbfeg*(Qadbfce+Qadbceg+Qadbfcg+Qacbfeg+Qadcfeg+Qbfcdeg-Qadbfeg)+
               adbgef*(Qadbcef+Qadbgce+Qadbgcf+Qacbgef+Qadcgef+Qbgcdef-Qadbgef)+
               aebdfg*(Qaebdcf+Qaebdcg+Qacbdfg+Qaebcfg+Qaecdfg+Qbdcefg-Qaebdfg)+
               aebfdg*(Qaebfcd+Qaebcdg+Qacbfdg+Qaebfcg+Qaecfdg+Qbfcedg-Qaebfdg)+
               aebgdf*(Qaebcdf+Qaebgcd+Qacbgdf+Qaebgcf+Qaecgdf+Qbgcedf-Qaebgdf)+
               afbdeg*(Qafbdce+Qacbdeg+Qafbdcg+Qafbceg+Qafcdeg+Qbdcfeg-Qafbdeg)+
               afbedg*(Qafbecd+Qacbedg+Qafbcdg+Qafbecg+Qafcedg+Qbecfdg-Qafbedg)+
               afbgde*(Qafbcde+Qacbgde+Qafbgcd+Qafbgce+Qafcgde+Qbgcfde-Qafbgde)+
               agbdef*(Qacbdef+Qagbdce+Qagbdcf+Qagbcef+Qagcdef+Qbdcgef-Qagbdef)+
               agbedf*(Qacbedf+Qagbecd+Qagbcdf+Qagbecf+Qagcedf+Qbecgdf-Qagbedf)+
               agbfde*(Qacbfde+Qagbcde+Qagbfcd+Qagbfce+Qagcfde+Qbfcgde-Qagbfde)+
               acdefg*(Qacbfde+Qacbgde+Qacbdfg+Qacbefg+Qabdefg+Qbcdefg-Qacdefg)+
               acdfeg*(Qacbedf+Qacbdeg+Qacbgdf+Qacbfeg+Qabdfeg+Qbcdfeg-Qacdfeg)+
               acdgef*(Qacbdef+Qacbedg+Qacbfdg+Qacbgef+Qabdgef+Qbcdgef-Qacdgef)+
               adcefg*(Qadbfce+Qadbgce+Qadbcfg+Qabcefg+Qadbefg+Qbdcefg-Qadcefg)+
               adcfeg*(Qadbecf+Qadbceg+Qadbgcf+Qabcfeg+Qadbfeg+Qbdcfeg-Qadcfeg)+
               adcgef*(Qadbcef+Qadbecg+Qadbfcg+Qabcgef+Qadbgef+Qbdcgef-Qadcgef)+
               aecdfg*(Qaebfcd+Qaebgcd+Qabcdfg+Qaebcfg+Qaebdfg+Qbecdfg-Qaecdfg)+
               aecfdg*(Qaebdcf+Qaebcdg+Qabcfdg+Qaebgcf+Qaebfdg+Qbecfdg-Qaecfdg)+
               aecgdf*(Qaebcdf+Qaebdcg+Qabcgdf+Qaebfcg+Qaebgdf+Qbecgdf-Qaecgdf)+
               afcdeg*(Qafbecd+Qabcdeg+Qafbgcd+Qafbceg+Qafbdeg+Qbfcdeg-Qafcdeg)+
               afcedg*(Qafbdce+Qabcedg+Qafbcdg+Qafbgce+Qafbedg+Qbfcedg-Qafcedg)+
               afcgde*(Qafbcde+Qabcgde+Qafbdcg+Qafbecg+Qafbgde+Qbfcgde-Qafcgde)+
               agcdef*(Qabcdef+Qagbecd+Qagbfcd+Qagbcef+Qagbdef+Qbgcdef-Qagcdef)+
               agcedf*(Qabcedf+Qagbdce+Qagbcdf+Qagbfce+Qagbedf+Qbgcedf-Qagcedf)+
               agcfde*(Qabcfde+Qagbcde+Qagbdcf+Qagbecf+Qagbfde+Qbgcfde-Qagcfde)+
               bcdefg*(Qafbcde+Qagbcde+Qadbcfg+Qaebcfg+Qabdefg+Qacdefg-Qbcdefg)+
               bcdfeg*(Qaebcdf+Qadbceg+Qagbcdf+Qafbceg+Qabdfeg+Qacdfeg-Qbcdfeg)+
               bcdgef*(Qadbcef+Qaebcdg+Qafbcdg+Qagbcef+Qabdgef+Qacdgef-Qbcdgef)+
               bdcefg*(Qafbdce+Qagbdce+Qacbdfg+Qabcefg+Qaebdfg+Qadcefg-Qbdcefg)+
               bdcfeg*(Qaebdcf+Qacbdeg+Qagbdcf+Qabcfeg+Qafbdeg+Qadcfeg-Qbdcfeg)+
               bdcgef*(Qacbdef+Qaebdcg+Qafbdcg+Qabcgef+Qagbdef+Qadcgef-Qbdcgef)+
               becdfg*(Qafbecd+Qagbecd+Qabcdfg+Qacbefg+Qadbefg+Qaecdfg-Qbecdfg)+
               becfdg*(Qadbecf+Qacbedg+Qabcfdg+Qagbecf+Qafbedg+Qaecfdg-Qbecfdg)+
               becgdf*(Qacbedf+Qadbecg+Qabcgdf+Qafbecg+Qagbedf+Qaecgdf-Qbecgdf)+
               bfcdeg*(Qaebfcd+Qabcdeg+Qagbfcd+Qacbfeg+Qadbfeg+Qafcdeg-Qbfcdeg)+
               bfcedg*(Qadbfce+Qabcedg+Qacbfdg+Qagbfce+Qaebfdg+Qafcedg-Qbfcedg)+
               bfcgde*(Qacbfde+Qabcgde+Qadbfcg+Qaebfcg+Qagbfde+Qafcgde-Qbfcgde)+
               bgcdef*(Qabcdef+Qaebgcd+Qafbgcd+Qacbgef+Qadbgef+Qagcdef-Qbgcdef)+
               bgcedf*(Qabcedf+Qadbgce+Qacbgdf+Qafbgce+Qaebgdf+Qagcedf-Qbgcedf)+
               bgcfde*(Qabcfde+Qacbgde+Qadbgcf+Qaebgcf+Qafbgde+Qagcfde-Qbgcfde))

        -4*(ab*ac*bc*(de*fg*(df+eg+dg+ef-de-fg)+df*eg*(de+fg+dg+ef-df-eg)+dg*ef*(de+fg+df+eg-dg-ef)-(de*df*ef+de*dg*eg+df*dg*fg+ef*eg*fg))+
            ab*ad*bd*(ce*fg*(cf+eg+cg+ef-ce-fg)+cf*eg*(ce+fg+cg+ef-cf-eg)+cg*ef*(ce+fg+cf+eg-cg-ef)-(ce*cf*ef+ce*cg*eg+cf*cg*fg+ef*eg*fg))+
            ab*ae*be*(cd*fg*(cf+dg+cg+df-cd-fg)+cf*dg*(cd+fg+cg+df-cf-dg)+cg*df*(cd+fg+cf+dg-cg-df)-(cd*cf*df+cd*cg*dg+cf*cg*fg+df*dg*fg))+
            ab*af*bf*(cd*eg*(ce+dg+cg+de-cd-eg)+ce*dg*(cd+eg+cg+de-ce-dg)+cg*de*(cd+eg+ce+dg-cg-de)-(cd*ce*de+cd*cg*dg+ce*cg*eg+de*dg*eg))+
            ab*ag*bg*(cd*ef*(ce+df+cf+de-cd-ef)+ce*df*(cd+ef+cf+de-ce-df)+cf*de*(cd+ef+ce+df-cf-de)-(cd*ce*de+cd*cf*df+ce*cf*ef+de*df*ef))+
            ac*ad*cd*(be*fg*(bf+eg+bg+ef-be-fg)+bf*eg*(be+fg+bg+ef-bf-eg)+bg*ef*(be+fg+bf+eg-bg-ef)-(be*bf*ef+be*bg*eg+bf*bg*fg+ef*eg*fg))+
            ac*ae*ce*(bd*fg*(bf+dg+bg+df-bd-fg)+bf*dg*(bd+fg+bg+df-bf-dg)+bg*df*(bd+fg+bf+dg-bg-df)-(bd*bf*df+bd*bg*dg+bf*bg*fg+df*dg*fg))+
            ac*af*cf*(bd*eg*(be+dg+bg+de-bd-eg)+be*dg*(bd+eg+bg+de-be-dg)+bg*de*(bd+eg+be+dg-bg-de)-(bd*be*de+bd*bg*dg+be*bg*eg+de*dg*eg))+
            ac*ag*cg*(bd*ef*(be+df+bf+de-bd-ef)+be*df*(bd+ef+bf+de-be-df)+bf*de*(bd+ef+be+df-bf-de)-(bd*be*de+bd*bf*df+be*bf*ef+de*df*ef))+
            ad*ae*de*(bc*fg*(bf+cg+bg+cf-bc-fg)+bf*cg*(bc+fg+bg+cf-bf-cg)+bg*cf*(bc+fg+bf+cg-bg-cf)-(bc*bf*cf+bc*bg*cg+bf*bg*fg+cf*cg*fg))+
            ad*af*df*(bc*eg*(be+cg+bg+ce-bc-eg)+be*cg*(bc+eg+bg+ce-be-cg)+bg*ce*(bc+eg+be+cg-bg-ce)-(bc*be*ce+bc*bg*cg+be*bg*eg+ce*cg*eg))+
            ad*ag*dg*(bc*ef*(be+cf+bf+ce-bc-ef)+be*cf*(bc+ef+bf+ce-be-cf)+bf*ce*(bc+ef+be+cf-bf-ce)-(bc*be*ce+bc*bf*cf+be*bf*ef+ce*cf*ef))+
            ae*af*ef*(bc*dg*(bd+cg+bg+cd-bc-dg)+bd*cg*(bc+dg+bg+cd-bd-cg)+bg*cd*(bc+dg+bd+cg-bg-cd)-(bc*bd*cd+bc*bg*cg+bd*bg*dg+cd*cg*dg))+
            ae*ag*eg*(bc*df*(bd+cf+bf+cd-bc-df)+bd*cf*(bc+df+bf+cd-bd-cf)+bf*cd*(bc+df+bd+cf-bf-cd)-(bc*bd*cd+bc*bf*cf+bd*bf*df+cd*cf*df))+
            af*ag*fg*(bc*de*(bd+ce+be+cd-bc-de)+bd*ce*(bc+de+be+cd-bd-ce)+be*cd*(bc+de+bd+ce-be-cd)-(bc*bd*cd+bc*be*ce+bd*be*de+cd*ce*de))+
            bc*bd*cd*(ae*fg*(af+eg+ag+ef-ae-fg)+af*eg*(ae+fg+ag+ef-af-eg)+ag*ef*(ae+fg+af+eg-ag-ef)-ef*eg*fg)+
            bc*be*ce*(ad*fg*(af+dg+ag+df-ad-fg)+af*dg*(ad+fg+ag+df-af-dg)+ag*df*(ad+fg+af+dg-ag-df)-df*dg*fg)+
            bc*bf*cf*(ad*eg*(ae+dg+ag+de-ad-eg)+ae*dg*(ad+eg+ag+de-ae-dg)+ag*de*(ad+eg+ae+dg-ag-de)-de*dg*eg)+
            bc*bg*cg*(ad*ef*(ae+df+af+de-ad-ef)+ae*df*(ad+ef+af+de-ae-df)+af*de*(ad+ef+ae+df-af-de)-de*df*ef)+
            bd*be*de*(ac*fg*(af+cg+ag+cf-ac-fg)+af*cg*(ac+fg+ag+cf-af-cg)+ag*cf*(ac+fg+af+cg-ag-cf)-cf*cg*fg)+
            bd*bf*df*(ac*eg*(ae+cg+ag+ce-ac-eg)+ae*cg*(ac+eg+ag+ce-ae-cg)+ag*ce*(ac+eg+ae+cg-ag-ce)-ce*cg*eg)+
            bd*bg*dg*(ac*ef*(ae+cf+af+ce-ac-ef)+ae*cf*(ac+ef+af+ce-ae-cf)+af*ce*(ac+ef+ae+cf-af-ce)-ce*cf*ef)+
            be*bf*ef*(ac*dg*(ad+cg+ag+cd-ac-dg)+ad*cg*(ac+dg+ag+cd-ad-cg)+ag*cd*(ac+dg+ad+cg-ag-cd)-cd*cg*dg)+
            be*bg*eg*(ac*df*(ad+cf+af+cd-ac-df)+ad*cf*(ac+df+af+cd-ad-cf)+af*cd*(ac+df+ad+cf-af-cd)-cd*cf*df)+
            bf*bg*fg*(ac*de*(ad+ce+ae+cd-ac-de)+ad*ce*(ac+de+ae+cd-ad-ce)+ae*cd*(ac+de+ad+ce-ae-cd)-cd*ce*de)+
            cd*ce*de*(ab*fg*(af+bg+ag+bf-ab-fg)+af*bg*(ab+fg+ag+bf-af-bg)+ag*bf*(ab+fg+af+bg-ag-bf))+
            cd*cf*df*(ab*eg*(ae+bg+ag+be-ab-eg)+ae*bg*(ab+eg+ag+be-ae-bg)+ag*be*(ab+eg+ae+bg-ag-be))+
            cd*cg*dg*(ab*ef*(ae+bf+af+be-ab-ef)+ae*bf*(ab+ef+af+be-ae-bf)+af*be*(ab+ef+ae+bf-af-be))+
            ce*cf*ef*(ab*dg*(ad+bg+ag+bd-ab-dg)+ad*bg*(ab+dg+ag+bd-ad-bg)+ag*bd*(ab+dg+ad+bg-ag-bd))+
            ce*cg*eg*(ab*df*(ad+bf+af+bd-ab-df)+ad*bf*(ab+df+af+bd-ad-bf)+af*bd*(ab+df+ad+bf-af-bd))+
            cf*cg*fg*(ab*de*(ad+be+ae+bd-ab-de)+ad*be*(ab+de+ae+bd-ad-be)+ae*bd*(ab+de+ad+be-ae-bd))+
            de*df*ef*(ab*cg*(ac+bg+ag+bc-ab-cg)+ac*bg*(ab+cg+ag+bc-ac-bg)+ag*bc*(ab+cg+ac+bg-ag-bc))+
            de*dg*eg*(ab*cf*(ac+bf+af+bc-ab-cf)+ac*bf*(ab+cf+af+bc-ac-bf)+af*bc*(ab+cf+ac+bf-af-bc))+
            df*dg*fg*(ab*ce*(ac+be+ae+bc-ab-ce)+ac*be*(ab+ce+ae+bc-ac-be)+ae*bc*(ab+ce+ac+be-ae-bc))+
            ef*eg*fg*(ab*cd*(ac+bd+ad+bc-ab-cd)+ac*bd*(ab+cd+ad+bc-ac-bd)+ad*bc*(ab+cd+ac+bd-ad-bc)))

        -4*(abcd*(aefg*(bfeg+bgef)+afeg*(befg+bgef)+agef*(befg+bfeg)+cefg*(dfeg+dgef)+cfeg*(defg+dgef)+cgef*(defg+dfeg))+
            abce*(adfg*(bfdg+bgdf)+afdg*(bdfg+bgdf)+agdf*(bdfg+bfdg)+cfdg*(defg+dfeg)+cgdf*(defg+dgef)+cgde*(dfeg+dgef))+
            abcf*(adeg*(bedg+bgde)+aedg*(bdeg+bgde)+agde*(bdeg+bedg))+
            abcg*(adef*(bedf+bfde)+aedf*(bdef+bfde)+afde*(bdef+bedf))+
            abde*(acfg*(bfcg+bgcf)+afcg*(bcfg+bgcf)+agcf*(bcfg+bfcg))+
            abdf*(aceg*(becg+bgce)+aecg*(bceg+bgce)+agce*(bceg+becg))+
            abdg*(acef*(becf+bfce)+aecf*(bcef+bfce)+afce*(bcef+becf))+
            abef*(acdg*(bdcg+bgcd)+adcg*(bcdg+bgcd)+agcd*(bcdg+bdcg))+
            abeg*(acdf*(bdcf+bfcd)+adcf*(bcdf+bfcd)+afcd*(bcdf+bdcf))+
            abfg*(acde*(bdce+becd)+adce*(bcde+becd)+aecd*(bcde+bdce))+
            acbd*(befg*(dfeg+dgef)+bfeg*(defg+dgef)+bgef*(defg+dfeg))+
            acbd*(aefg*(cfeg+cgef)+afeg*(cefg+cgef)+agef*(cefg+cfeg))+
            acbe*(adfg*(cfdg+cgdf)+afdg*(cdfg+cgdf)+agdf*(cdfg+cfdg)+bfdg*(defg+dfeg)+bgdf*(defg+dgef))+
            acbf*(adeg*(cedg+cgde)+aedg*(cdeg+cgde)+agde*(cdeg+cedg)+bgde*(dfeg+dgef))+
            acbg*(adef*(cedf+cfde)+aedf*(cdef+cfde)+afde*(cdef+cedf))+
            acde*(afbg*(bcfg+bfcg)+agbf*(bcfg+bgcf))+
            acdf*(aebg*(bceg+becg)+agbe*(bceg+bgce))+
            acdg*(aebf*(bcef+becf)+afbe*(bcef+bfce))+
            acef*(adbg*(bcdg+bdcg)+agbd*(bcdg+bgcd))+
            aceg*(adbf*(bcdf+bdcf)+afbd*(bcdf+bfcd))+
            acfg*(adbe*(bcde+bdce)+aebd*(bcde+becd))+
            adbc*(aefg*(dfeg+dgef)+afeg*(defg+dgef)+agef*(defg+dfeg)+befg*(cfeg+cgef)+bfeg*(cefg+cgef)+bgef*(cefg+cfeg))+
            adbe*(afcg*(cdfg+cfdg)+agcf*(cdfg+cgdf)+bfcg*(cefg+cfeg)+bgcf*(cefg+cgef))+
            adbf*(aecg*(cdeg+cedg)+agce*(cdeg+cgde)+bgce*(cfeg+cgef))+
            adbg*(aecf*(cdef+cedf)+afce*(cdef+cfde))+
            adce*(afbg*(bdfg+bfdg)+agbf*(bdfg+bgdf))+
            adcf*(aebg*(bdeg+bedg)+agbe*(bdeg+bgde))+
            adcg*(aebf*(bdef+bedf)+afbe*(bdef+bfde))+
            adef*(agbc*(bdcg+bgcd))+
            adeg*(afbc*(bdcf+bfcd))+
            adfg*(aebc*(bdce+becd))+
            aebc*(afdg*(defg+dfeg)+agdf*(defg+dgef)+bdfg*(cfdg+cgdf)+bfdg*(cdfg+cgdf)+bgdf*(cdfg+cfdg))+
            aebd*(afcg*(cefg+cfeg)+agcf*(cefg+cgef)+bfcg*(cdfg+cfdg)+bgcf*(cdfg+cgdf))+
            aebf*(agcd*(cedg+cgde)+bgcd*(cfdg+cgdf))+
            aebg*(afcd*(cedf+cfde))+
            aecd*(afbg*(befg+bfeg)+agbf*(befg+bgef))+
            aecf*(agbd*(bedg+bgde))+
            aecg*(afbd*(bedf+bfde))+
            aedf*(agbc*(becg+bgce))+
            aedg*(afbc*(becf+bfce))+
            afbc*(agde*(dfeg+dgef)+bdeg*(cedg+cgde)+bedg*(cdeg+cgde)+bgde*(cdeg+cedg))+
            afbd*(agce*(cfeg+cgef)+becg*(cdeg+cedg)+bgce*(cdeg+cgde))+
            afbe*(agcd*(cfdg+cgdf)+bgcd*(cedg+cgde))+
            afcd*(agbe*(bfeg+bgef))+
            afce*(agbd*(bfdg+bgdf))+
            afde*(agbc*(bfcg+bgcf))+
            agbc*(bdef*(cedf+cfde)+bedf*(cdef+cfde)+bfde*(cdef+cedf))+
            agbd*(becf*(cdef+cedf)+bfce*(cdef+cfde))+
            agbe*(bfcd*(cedf+cfde)));
    }

DEVICE inline bool seq2(OverlapReal as,OverlapReal bs,
              OverlapReal ar,OverlapReal br,
              OverlapReal ab)
    {
        if(as*(ab+br-ar) < OverlapReal(-EPS)) return false;
        if(bs*(ab+ar-br) < OverlapReal(-EPS)) return false;

        return(ab*(ar+br-ab)+ar*(ab+br-ar)+br*(ab+ar-br) <= 0);
    }

DEVICE inline bool seq3(OverlapReal as,OverlapReal bs,OverlapReal cs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,
              OverlapReal ab,OverlapReal ac,
              OverlapReal bc)
    {
        if(as*(bc*(ab+ac-bc+br+cr-ar-ar)-(ab-ac)*(br-cr)) < OverlapReal(-EPS)) return false;
        if(bs*(ac*(ab+bc-ac+ar+cr-br-br)-(ab-bc)*(ar-cr)) < OverlapReal(-EPS)) return false;
        if(cs*(ab*(ac+bc-ab+ar+br-cr-cr)-(ac-bc)*(ar-br)) < OverlapReal(-EPS)) return false;

        return(2*(bc*ar*(ab+ac-bc+br+cr-ar)+
                  ac*br*(ab+bc-ac+ar+cr-br)+
                  ab*cr*(ac+bc-ab+ar+br-cr))
               -(bc+ar)*(ac+br)*(ab+cr)
               -(bc-ar)*(ac-br)*(ab-cr) <= OverlapReal(EPS));
    }

DEVICE inline bool seq4(OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,
              OverlapReal bc,OverlapReal bd,
              OverlapReal cd)
    {
        if(as*((bc*(ad+dr-ar)-(ab-ac)*(br-cr))*(bd+cd-bc)+
               (bd*(ac+cr-ar)-(ab-ad)*(br-dr))*(bc+cd-bd)+
               (cd*(ab+br-ar)-(ac-ad)*(cr-dr))*(bc+bd-cd)-2*bc*bd*cd) < OverlapReal(-EPS)) return false;
        if(bs*((ac*(bd+dr-br)-(ab-bc)*(ar-cr))*(ad+cd-ac)+
               (ad*(bc+cr-br)-(ab-bd)*(ar-dr))*(ac+cd-ad)+
               (cd*(ab+ar-br)-(bc-bd)*(cr-dr))*(ac+ad-cd)-2*ac*ad*cd) < OverlapReal(-EPS)) return false;
        if(cs*((ab*(cd+dr-cr)-(ac-bc)*(ar-br))*(ad+bd-ab)+
               (ad*(bc+br-cr)-(ac-cd)*(ar-dr))*(ab+bd-ad)+
               (bd*(ac+ar-cr)-(bc-cd)*(br-dr))*(ab+ad-bd)-2*ab*ad*bd) < OverlapReal(-EPS)) return false;
        if(ds*((ab*(cd+cr-dr)-(ad-bd)*(ar-br))*(ac+bc-ab)+
               (ac*(bd+br-dr)-(ad-cd)*(ar-cr))*(ab+bc-ac)+
               (bc*(ad+ar-dr)-(bd-cd)*(br-cr))*(ab+ac-bc)-2*ab*ac*bc) < OverlapReal(-EPS)) return false;

        return(vok4(ab,ac,ad,ar,bc,bd,br,cd,cr,dr) <= OverlapReal(EPS));
    }

DEVICE inline bool seq5(OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
              OverlapReal bc,OverlapReal bd,OverlapReal be,
              OverlapReal cd,OverlapReal ce,
              OverlapReal de)
    {
        if(as*ang4(bc,bd,be,ab,br,cd,ce,ac,cr,de,ad,dr,ae,er,ar) < OverlapReal(-EPS)) return false;
        if(bs*ang4(ac,ad,ae,ab,ar,cd,ce,bc,cr,de,bd,dr,be,er,br) < OverlapReal(-EPS)) return false;
        if(cs*ang4(ab,ad,ae,ac,ar,bd,be,bc,br,de,cd,dr,ce,er,cr) < OverlapReal(-EPS)) return false;
        if(ds*ang4(ab,ac,ae,ad,ar,bc,be,bd,br,ce,cd,cr,de,er,dr) < OverlapReal(-EPS)) return false;
        if(es*ang4(ab,ac,ad,ae,ar,bc,bd,be,br,cd,ce,cr,de,dr,er) < OverlapReal(-EPS)) return false;

        return(vok5(ab,ac,ad,ae,ar,bc,bd,be,br,cd,ce,cr,de,dr,er) <= OverlapReal(EPS));
    }

DEVICE inline bool seq6(OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,OverlapReal fs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,OverlapReal fr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
              OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
              OverlapReal cd,OverlapReal ce,OverlapReal cf,
              OverlapReal de,OverlapReal df,
              OverlapReal ef)
    {
        if(as*ang5(bc,bd,be,bf,ab,br,cd,ce,cf,ac,cr,de,df,ad,dr,ef,ae,er,af,fr,ar) < OverlapReal(-EPS)) return false;
        if(bs*ang5(ac,ad,ae,af,ab,ar,cd,ce,cf,bc,cr,de,df,bd,dr,ef,be,er,bf,fr,br) < OverlapReal(-EPS)) return false;
        if(cs*ang5(ab,ad,ae,af,ac,ar,bd,be,bf,bc,br,de,df,cd,dr,ef,ce,er,cf,fr,cr) < OverlapReal(-EPS)) return false;
        if(ds*ang5(ab,ac,ae,af,ad,ar,bc,be,bf,bd,br,ce,cf,cd,cr,ef,de,er,df,fr,dr) < OverlapReal(-EPS)) return false;
        if(es*ang5(ab,ac,ad,af,ae,ar,bc,bd,bf,be,br,cd,cf,ce,cr,df,de,dr,ef,fr,er) < OverlapReal(-EPS)) return false;
        if(fs*ang5(ab,ac,ad,ae,af,ar,bc,bd,be,bf,br,cd,ce,cf,cr,de,df,dr,ef,er,fr) < OverlapReal(-EPS)) return false;

        return(vok6(ab,ac,ad,ae,af,ar,bc,bd,be,bf,br,cd,ce,cf,cr,de,df,dr,ef,er,fr) <= OverlapReal(EPS));
    }

DEVICE inline bool sep2(bool convex,
              OverlapReal as,OverlapReal bs,
              OverlapReal ar,OverlapReal br,
              OverlapReal ab)
    {
        if(convex && (DIM == 0)) return false;

        return seq2(as,bs,ar,br,ab);
    }

DEVICE inline bool sep3(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,
              OverlapReal ab,OverlapReal ac,
              OverlapReal bc)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;

        if(convex && (DIM == 1)) return false;

        return seq3(as,bs,cs,ar,br,cr,ab,ac,bc);
    }

DEVICE inline bool sep4(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,
              OverlapReal bc,OverlapReal bd,
              OverlapReal cd)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(as,ds,ar,dr,ad)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;
        if(seq2(bs,ds,br,dr,bd)) return true;
        if(seq2(cs,ds,cr,dr,cd)) return true;

        if(convex && (DIM == 1)) return false;

        if(seq3(as,bs,cs,ar,br,cr,ab,ac,bc)) return true;
        if(seq3(as,bs,ds,ar,br,dr,ab,ad,bd)) return true;
        if(seq3(as,cs,ds,ar,cr,dr,ac,ad,cd)) return true;
        if(seq3(bs,cs,ds,br,cr,dr,bc,bd,cd)) return true;

        if(convex && (DIM == 2)) return false;

        return seq4(as,bs,cs,ds,ar,br,cr,dr,ab,ac,ad,bc,bd,cd);
    }

DEVICE inline bool sep5(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
              OverlapReal bc,OverlapReal bd,OverlapReal be,
              OverlapReal cd,OverlapReal ce,
              OverlapReal de)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(as,ds,ar,dr,ad)) return true;
        if(seq2(as,es,ar,er,ae)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;
        if(seq2(bs,ds,br,dr,bd)) return true;
        if(seq2(bs,es,br,er,be)) return true;
        if(seq2(cs,ds,cr,dr,cd)) return true;
        if(seq2(cs,es,cr,er,ce)) return true;
        if(seq2(ds,es,dr,er,de)) return true;

        if(convex && (DIM == 1)) return false;

        if(seq3(as,bs,cs,ar,br,cr,ab,ac,bc)) return true;
        if(seq3(as,bs,ds,ar,br,dr,ab,ad,bd)) return true;
        if(seq3(as,bs,es,ar,br,er,ab,ae,be)) return true;
        if(seq3(as,cs,ds,ar,cr,dr,ac,ad,cd)) return true;
        if(seq3(as,cs,es,ar,cr,er,ac,ae,ce)) return true;
        if(seq3(as,ds,es,ar,dr,er,ad,ae,de)) return true;
        if(seq3(bs,cs,ds,br,cr,dr,bc,bd,cd)) return true;
        if(seq3(bs,cs,es,br,cr,er,bc,be,ce)) return true;
        if(seq3(bs,ds,es,br,dr,er,bd,be,de)) return true;
        if(seq3(cs,ds,es,cr,dr,er,cd,ce,de)) return true;

        if(convex && (DIM == 2)) return false;

        if(seq4(as,bs,cs,ds,ar,br,cr,dr,ab,ac,ad,bc,bd,cd)) return true;
        if(seq4(as,bs,cs,es,ar,br,cr,er,ab,ac,ae,bc,be,ce)) return true;
        if(seq4(as,bs,ds,es,ar,br,dr,er,ab,ad,ae,bd,be,de)) return true;
        if(seq4(as,cs,ds,es,ar,cr,dr,er,ac,ad,ae,cd,ce,de)) return true;
        if(seq4(bs,cs,ds,es,br,cr,dr,er,bc,bd,be,cd,ce,de)) return true;

        if(convex && (DIM == 3)) return false;

        return seq5(as,bs,cs,ds,es,ar,br,cr,dr,er,ab,ac,ad,ae,bc,bd,be,cd,ce,de);
    }

DEVICE inline bool sep6(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,OverlapReal fs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,OverlapReal fr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
              OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
              OverlapReal cd,OverlapReal ce,OverlapReal cf,
              OverlapReal de,OverlapReal df,
              OverlapReal ef)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(as,ds,ar,dr,ad)) return true;
        if(seq2(as,es,ar,er,ae)) return true;
        if(seq2(as,fs,ar,fr,af)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;
        if(seq2(bs,ds,br,dr,bd)) return true;
        if(seq2(bs,es,br,er,be)) return true;
        if(seq2(bs,fs,br,fr,bf)) return true;
        if(seq2(cs,ds,cr,dr,cd)) return true;
        if(seq2(cs,es,cr,er,ce)) return true;
        if(seq2(cs,fs,cr,fr,cf)) return true;
        if(seq2(ds,es,dr,er,de)) return true;
        if(seq2(ds,fs,dr,fr,df)) return true;
        if(seq2(es,fs,er,fr,ef)) return true;

        if(convex && (DIM == 1)) return false;

        if(seq3(as,bs,cs,ar,br,cr,ab,ac,bc)) return true;
        if(seq3(as,bs,ds,ar,br,dr,ab,ad,bd)) return true;
        if(seq3(as,bs,es,ar,br,er,ab,ae,be)) return true;
        if(seq3(as,bs,fs,ar,br,fr,ab,af,bf)) return true;
        if(seq3(as,cs,ds,ar,cr,dr,ac,ad,cd)) return true;
        if(seq3(as,cs,es,ar,cr,er,ac,ae,ce)) return true;
        if(seq3(as,cs,fs,ar,cr,fr,ac,af,cf)) return true;
        if(seq3(as,ds,es,ar,dr,er,ad,ae,de)) return true;
        if(seq3(as,ds,fs,ar,dr,fr,ad,af,df)) return true;
        if(seq3(as,es,fs,ar,er,fr,ae,af,ef)) return true;
        if(seq3(bs,cs,ds,br,cr,dr,bc,bd,cd)) return true;
        if(seq3(bs,cs,es,br,cr,er,bc,be,ce)) return true;
        if(seq3(bs,cs,fs,br,cr,fr,bc,bf,cf)) return true;
        if(seq3(bs,ds,es,br,dr,er,bd,be,de)) return true;
        if(seq3(bs,ds,fs,br,dr,fr,bd,bf,df)) return true;
        if(seq3(bs,es,fs,br,er,fr,be,bf,ef)) return true;
        if(seq3(cs,ds,es,cr,dr,er,cd,ce,de)) return true;
        if(seq3(cs,ds,fs,cr,dr,fr,cd,cf,df)) return true;
        if(seq3(cs,es,fs,cr,er,fr,ce,cf,ef)) return true;
        if(seq3(ds,es,fs,dr,er,fr,de,df,ef)) return true;

        if(convex && (DIM == 2)) return false;

        if(seq4(as,bs,cs,ds,ar,br,cr,dr,ab,ac,ad,bc,bd,cd)) return true;
        if(seq4(as,bs,cs,es,ar,br,cr,er,ab,ac,ae,bc,be,ce)) return true;
        if(seq4(as,bs,cs,fs,ar,br,cr,fr,ab,ac,af,bc,bf,cf)) return true;
        if(seq4(as,bs,ds,es,ar,br,dr,er,ab,ad,ae,bd,be,de)) return true;
        if(seq4(as,bs,ds,fs,ar,br,dr,fr,ab,ad,af,bd,bf,df)) return true;
        if(seq4(as,bs,es,fs,ar,br,er,fr,ab,ae,af,be,bf,ef)) return true;
        if(seq4(as,cs,ds,es,ar,cr,dr,er,ac,ad,ae,cd,ce,de)) return true;
        if(seq4(as,cs,ds,fs,ar,cr,dr,fr,ac,ad,af,cd,cf,df)) return true;
        if(seq4(as,cs,es,fs,ar,cr,er,fr,ac,ae,af,ce,cf,ef)) return true;
        if(seq4(as,ds,es,fs,ar,dr,er,fr,ad,ae,af,de,df,ef)) return true;
        if(seq4(bs,cs,ds,es,br,cr,dr,er,bc,bd,be,cd,ce,de)) return true;
        if(seq4(bs,cs,ds,fs,br,cr,dr,fr,bc,bd,bf,cd,cf,df)) return true;
        if(seq4(bs,cs,es,fs,br,cr,er,fr,bc,be,bf,ce,cf,ef)) return true;
        if(seq4(bs,ds,es,fs,br,dr,er,fr,bd,be,bf,de,df,ef)) return true;
        if(seq4(cs,ds,es,fs,cr,dr,er,fr,cd,ce,cf,de,df,ef)) return true;

        if(convex && (DIM == 3)) return false;

        if(seq5(as,bs,cs,ds,es,ar,br,cr,dr,er,ab,ac,ad,ae,bc,bd,be,cd,ce,de)) return true;
        if(seq5(as,bs,cs,ds,fs,ar,br,cr,dr,fr,ab,ac,ad,af,bc,bd,bf,cd,cf,df)) return true;
        if(seq5(as,bs,cs,es,fs,ar,br,cr,er,fr,ab,ac,ae,af,bc,be,bf,ce,cf,ef)) return true;
        if(seq5(as,bs,ds,es,fs,ar,br,dr,er,fr,ab,ad,ae,af,bd,be,bf,de,df,ef)) return true;
        if(seq5(as,cs,ds,es,fs,ar,cr,dr,er,fr,ac,ad,ae,af,cd,ce,cf,de,df,ef)) return true;
        if(seq5(bs,cs,ds,es,fs,br,cr,dr,er,fr,bc,bd,be,bf,cd,ce,cf,de,df,ef)) return true;

        if(convex && (DIM == 4)) return false;

        return seq6(as,bs,cs,ds,es,fs,ar,br,cr,dr,er,fr,ab,ac,ad,ae,af,bc,bd,be,bf,cd,ce,cf,de,df,ef);
    }

DEVICE inline bool sep7(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,OverlapReal fs,OverlapReal gs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,OverlapReal fr,OverlapReal gr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,
              OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,
              OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,
              OverlapReal de,OverlapReal df,OverlapReal dg,
              OverlapReal ef,OverlapReal eg,
              OverlapReal fg)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(as,ds,ar,dr,ad)) return true;
        if(seq2(as,es,ar,er,ae)) return true;
        if(seq2(as,fs,ar,fr,af)) return true;
        if(seq2(as,gs,ar,gr,ag)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;
        if(seq2(bs,ds,br,dr,bd)) return true;
        if(seq2(bs,es,br,er,be)) return true;
        if(seq2(bs,fs,br,fr,bf)) return true;
        if(seq2(bs,gs,br,gr,bg)) return true;
        if(seq2(cs,ds,cr,dr,cd)) return true;
        if(seq2(cs,es,cr,er,ce)) return true;
        if(seq2(cs,fs,cr,fr,cf)) return true;
        if(seq2(cs,gs,cr,gr,cg)) return true;
        if(seq2(ds,es,dr,er,de)) return true;
        if(seq2(ds,fs,dr,fr,df)) return true;
        if(seq2(ds,gs,dr,gr,dg)) return true;
        if(seq2(es,fs,er,fr,ef)) return true;
        if(seq2(es,gs,er,gr,eg)) return true;
        if(seq2(fs,gs,fr,gr,fg)) return true;

        if(convex && (DIM == 1)) return false;

        if(seq3(as,bs,cs,ar,br,cr,ab,ac,bc)) return true;
        if(seq3(as,bs,ds,ar,br,dr,ab,ad,bd)) return true;
        if(seq3(as,bs,es,ar,br,er,ab,ae,be)) return true;
        if(seq3(as,bs,fs,ar,br,fr,ab,af,bf)) return true;
        if(seq3(as,bs,gs,ar,br,gr,ab,ag,bg)) return true;
        if(seq3(as,cs,ds,ar,cr,dr,ac,ad,cd)) return true;
        if(seq3(as,cs,es,ar,cr,er,ac,ae,ce)) return true;
        if(seq3(as,cs,fs,ar,cr,fr,ac,af,cf)) return true;
        if(seq3(as,cs,gs,ar,cr,gr,ac,ag,cg)) return true;
        if(seq3(as,ds,es,ar,dr,er,ad,ae,de)) return true;
        if(seq3(as,ds,fs,ar,dr,fr,ad,af,df)) return true;
        if(seq3(as,ds,gs,ar,dr,gr,ad,ag,dg)) return true;
        if(seq3(as,es,fs,ar,er,fr,ae,af,ef)) return true;
        if(seq3(as,es,gs,ar,er,gr,ae,ag,eg)) return true;
        if(seq3(as,fs,gs,ar,fr,gr,af,ag,fg)) return true;
        if(seq3(bs,cs,ds,br,cr,dr,bc,bd,cd)) return true;
        if(seq3(bs,cs,es,br,cr,er,bc,be,ce)) return true;
        if(seq3(bs,cs,fs,br,cr,fr,bc,bf,cf)) return true;
        if(seq3(bs,cs,gs,br,cr,gr,bc,bg,cg)) return true;
        if(seq3(bs,ds,es,br,dr,er,bd,be,de)) return true;
        if(seq3(bs,ds,fs,br,dr,fr,bd,bf,df)) return true;
        if(seq3(bs,ds,gs,br,dr,gr,bd,bg,dg)) return true;
        if(seq3(bs,es,fs,br,er,fr,be,bf,ef)) return true;
        if(seq3(bs,es,gs,br,er,gr,be,bg,eg)) return true;
        if(seq3(bs,fs,gs,br,fr,gr,bf,bg,fg)) return true;
        if(seq3(cs,ds,es,cr,dr,er,cd,ce,de)) return true;
        if(seq3(cs,ds,fs,cr,dr,fr,cd,cf,df)) return true;
        if(seq3(cs,ds,gs,cr,dr,gr,cd,cg,dg)) return true;
        if(seq3(cs,es,fs,cr,er,fr,ce,cf,ef)) return true;
        if(seq3(cs,es,gs,cr,er,gr,ce,cg,eg)) return true;
        if(seq3(cs,fs,gs,cr,fr,gr,cf,cg,fg)) return true;
        if(seq3(ds,es,fs,dr,er,fr,de,df,ef)) return true;
        if(seq3(ds,es,gs,dr,er,gr,de,dg,eg)) return true;
        if(seq3(ds,fs,gs,dr,fr,gr,df,dg,fg)) return true;
        if(seq3(es,fs,gs,er,fr,gr,ef,eg,fg)) return true;

        if(convex && (DIM == 2)) return false;

        if(seq4(as,bs,cs,ds,ar,br,cr,dr,ab,ac,ad,bc,bd,cd)) return true;
        if(seq4(as,bs,cs,es,ar,br,cr,er,ab,ac,ae,bc,be,ce)) return true;
        if(seq4(as,bs,cs,fs,ar,br,cr,fr,ab,ac,af,bc,bf,cf)) return true;
        if(seq4(as,bs,cs,gs,ar,br,cr,gr,ab,ac,ag,bc,bg,cg)) return true;
        if(seq4(as,bs,ds,es,ar,br,dr,er,ab,ad,ae,bd,be,de)) return true;
        if(seq4(as,bs,ds,fs,ar,br,dr,fr,ab,ad,af,bd,bf,df)) return true;
        if(seq4(as,bs,ds,gs,ar,br,dr,gr,ab,ad,ag,bd,bg,dg)) return true;
        if(seq4(as,bs,es,fs,ar,br,er,fr,ab,ae,af,be,bf,ef)) return true;
        if(seq4(as,bs,es,gs,ar,br,er,gr,ab,ae,ag,be,bg,eg)) return true;
        if(seq4(as,bs,fs,gs,ar,br,fr,gr,ab,af,ag,bf,bg,fg)) return true;
        if(seq4(as,cs,ds,es,ar,cr,dr,er,ac,ad,ae,cd,ce,de)) return true;
        if(seq4(as,cs,ds,fs,ar,cr,dr,fr,ac,ad,af,cd,cf,df)) return true;
        if(seq4(as,cs,ds,gs,ar,cr,dr,gr,ac,ad,ag,cd,cg,dg)) return true;
        if(seq4(as,cs,es,fs,ar,cr,er,fr,ac,ae,af,ce,cf,ef)) return true;
        if(seq4(as,cs,es,gs,ar,cr,er,gr,ac,ae,ag,ce,cg,eg)) return true;
        if(seq4(as,cs,fs,gs,ar,cr,fr,gr,ac,af,ag,cf,cg,fg)) return true;
        if(seq4(as,ds,es,fs,ar,dr,er,fr,ad,ae,af,de,df,ef)) return true;
        if(seq4(as,ds,es,gs,ar,dr,er,gr,ad,ae,ag,de,dg,eg)) return true;
        if(seq4(as,ds,fs,gs,ar,dr,fr,gr,ad,af,ag,df,dg,fg)) return true;
        if(seq4(as,es,fs,gs,ar,er,fr,gr,ae,af,ag,ef,eg,fg)) return true;
        if(seq4(bs,cs,ds,es,br,cr,dr,er,bc,bd,be,cd,ce,de)) return true;
        if(seq4(bs,cs,ds,fs,br,cr,dr,fr,bc,bd,bf,cd,cf,df)) return true;
        if(seq4(bs,cs,ds,gs,br,cr,dr,gr,bc,bd,bg,cd,cg,dg)) return true;
        if(seq4(bs,cs,es,fs,br,cr,er,fr,bc,be,bf,ce,cf,ef)) return true;
        if(seq4(bs,cs,es,gs,br,cr,er,gr,bc,be,bg,ce,cg,eg)) return true;
        if(seq4(bs,cs,fs,gs,br,cr,fr,gr,bc,bf,bg,cf,cg,fg)) return true;
        if(seq4(bs,ds,es,fs,br,dr,er,fr,bd,be,bf,de,df,ef)) return true;
        if(seq4(bs,ds,es,gs,br,dr,er,gr,bd,be,bg,de,dg,eg)) return true;
        if(seq4(bs,ds,fs,gs,br,dr,fr,gr,bd,bf,bg,df,dg,fg)) return true;
        if(seq4(bs,es,fs,gs,br,er,fr,gr,be,bf,bg,ef,eg,fg)) return true;
        if(seq4(cs,ds,es,fs,cr,dr,er,fr,cd,ce,cf,de,df,ef)) return true;
        if(seq4(cs,ds,es,gs,cr,dr,er,gr,cd,ce,cg,de,dg,eg)) return true;
        if(seq4(cs,ds,fs,gs,cr,dr,fr,gr,cd,cf,cg,df,dg,fg)) return true;
        if(seq4(cs,es,fs,gs,cr,er,fr,gr,ce,cf,cg,ef,eg,fg)) return true;
        if(seq4(ds,es,fs,gs,dr,er,fr,gr,de,df,dg,ef,eg,fg)) return true;

        if(convex && (DIM == 3)) return false;

        if(seq5(as,bs,cs,ds,es,ar,br,cr,dr,er,ab,ac,ad,ae,bc,bd,be,cd,ce,de)) return true;
        if(seq5(as,bs,cs,ds,fs,ar,br,cr,dr,fr,ab,ac,ad,af,bc,bd,bf,cd,cf,df)) return true;
        if(seq5(as,bs,cs,ds,gs,ar,br,cr,dr,gr,ab,ac,ad,ag,bc,bd,bg,cd,cg,dg)) return true;
        if(seq5(as,bs,cs,es,fs,ar,br,cr,er,fr,ab,ac,ae,af,bc,be,bf,ce,cf,ef)) return true;
        if(seq5(as,bs,cs,es,gs,ar,br,cr,er,gr,ab,ac,ae,ag,bc,be,bg,ce,cg,eg)) return true;
        if(seq5(as,bs,cs,fs,gs,ar,br,cr,fr,gr,ab,ac,af,ag,bc,bf,bg,cf,cg,fg)) return true;
        if(seq5(as,bs,ds,es,fs,ar,br,dr,er,fr,ab,ad,ae,af,bd,be,bf,de,df,ef)) return true;
        if(seq5(as,bs,ds,es,gs,ar,br,dr,er,gr,ab,ad,ae,ag,bd,be,bg,de,dg,eg)) return true;
        if(seq5(as,bs,ds,fs,gs,ar,br,dr,fr,gr,ab,ad,af,ag,bd,bf,bg,df,dg,fg)) return true;
        if(seq5(as,bs,es,fs,gs,ar,br,er,fr,gr,ab,ae,af,ag,be,bf,bg,ef,eg,fg)) return true;
        if(seq5(as,cs,ds,es,fs,ar,cr,dr,er,fr,ac,ad,ae,af,cd,ce,cf,de,df,ef)) return true;
        if(seq5(as,cs,ds,es,gs,ar,cr,dr,er,gr,ac,ad,ae,ag,cd,ce,cg,de,dg,eg)) return true;
        if(seq5(as,cs,ds,fs,gs,ar,cr,dr,fr,gr,ac,ad,af,ag,cd,cf,cg,df,dg,fg)) return true;
        if(seq5(as,cs,es,fs,gs,ar,cr,er,fr,gr,ac,ae,af,ag,ce,cf,cg,ef,eg,fg)) return true;
        if(seq5(as,ds,es,fs,gs,ar,dr,er,fr,gr,ad,ae,af,ag,de,df,dg,ef,eg,fg)) return true;
        if(seq5(bs,cs,ds,es,fs,br,cr,dr,er,fr,bc,bd,be,bf,cd,ce,cf,de,df,ef)) return true;
        if(seq5(bs,cs,ds,es,gs,br,cr,dr,er,gr,bc,bd,be,bg,cd,ce,cg,de,dg,eg)) return true;
        if(seq5(bs,cs,ds,fs,gs,br,cr,dr,fr,gr,bc,bd,bf,bg,cd,cf,cg,df,dg,fg)) return true;
        if(seq5(bs,cs,es,fs,gs,br,cr,er,fr,gr,bc,be,bf,bg,ce,cf,cg,ef,eg,fg)) return true;
        if(seq5(bs,ds,es,fs,gs,br,dr,er,fr,gr,bd,be,bf,bg,de,df,dg,ef,eg,fg)) return true;
        if(seq5(cs,ds,es,fs,gs,cr,dr,er,fr,gr,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;

        if(convex && (DIM == 4)) return false;

        if(seq6(as,bs,cs,ds,es,fs,ar,br,cr,dr,er,fr,ab,ac,ad,ae,af,bc,bd,be,bf,cd,ce,cf,de,df,ef)) return true;
        if(seq6(as,bs,cs,ds,es,gs,ar,br,cr,dr,er,gr,ab,ac,ad,ae,ag,bc,bd,be,bg,cd,ce,cg,de,dg,eg)) return true;
        if(seq6(as,bs,cs,ds,fs,gs,ar,br,cr,dr,fr,gr,ab,ac,ad,af,ag,bc,bd,bf,bg,cd,cf,cg,df,dg,fg)) return true;
        if(seq6(as,bs,cs,es,fs,gs,ar,br,cr,er,fr,gr,ab,ac,ae,af,ag,bc,be,bf,bg,ce,cf,cg,ef,eg,fg)) return true;
        if(seq6(as,bs,ds,es,fs,gs,ar,br,dr,er,fr,gr,ab,ad,ae,af,ag,bd,be,bf,bg,de,df,dg,ef,eg,fg)) return true;
        if(seq6(as,cs,ds,es,fs,gs,ar,cr,dr,er,fr,gr,ac,ad,ae,af,ag,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;
        if(seq6(bs,cs,ds,es,fs,gs,br,cr,dr,er,fr,gr,bc,bd,be,bf,bg,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;

        if(convex && (DIM == 5)) return false;

        //return seq7(as,bs,cs,ds,es,fs,gs,ar,br,cr,dr,er,fr,gr,ab,ac,ad,ae,af,ag,bc,bd,be,bf,bg,cd,ce,cf,cg,de,df,dg,ef,eg,fg);
        return false;
    }

DEVICE inline bool sep8(bool convex,
              OverlapReal as,OverlapReal bs,OverlapReal cs,OverlapReal ds,OverlapReal es,OverlapReal fs,OverlapReal gs,OverlapReal hs,
              OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal er,OverlapReal fr,OverlapReal gr,OverlapReal hr,
              OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,OverlapReal ah,
              OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,OverlapReal bh,
              OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,OverlapReal ch,
              OverlapReal de,OverlapReal df,OverlapReal dg,OverlapReal dh,
              OverlapReal ef,OverlapReal eg,OverlapReal eh,
              OverlapReal fg,OverlapReal fh,
              OverlapReal gh)
    {
        if(convex && (DIM == 0)) return false;

        if(seq2(as,bs,ar,br,ab)) return true;
        if(seq2(as,cs,ar,cr,ac)) return true;
        if(seq2(as,ds,ar,dr,ad)) return true;
        if(seq2(as,es,ar,er,ae)) return true;
        if(seq2(as,fs,ar,fr,af)) return true;
        if(seq2(as,gs,ar,gr,ag)) return true;
        if(seq2(as,hs,ar,hr,ah)) return true;
        if(seq2(bs,cs,br,cr,bc)) return true;
        if(seq2(bs,ds,br,dr,bd)) return true;
        if(seq2(bs,es,br,er,be)) return true;
        if(seq2(bs,fs,br,fr,bf)) return true;
        if(seq2(bs,gs,br,gr,bg)) return true;
        if(seq2(bs,hs,br,hr,bh)) return true;
        if(seq2(cs,ds,cr,dr,cd)) return true;
        if(seq2(cs,es,cr,er,ce)) return true;
        if(seq2(cs,fs,cr,fr,cf)) return true;
        if(seq2(cs,gs,cr,gr,cg)) return true;
        if(seq2(cs,hs,cr,hr,ch)) return true;
        if(seq2(ds,es,dr,er,de)) return true;
        if(seq2(ds,fs,dr,fr,df)) return true;
        if(seq2(ds,gs,dr,gr,dg)) return true;
        if(seq2(ds,hs,dr,hr,dh)) return true;
        if(seq2(es,fs,er,fr,ef)) return true;
        if(seq2(es,gs,er,gr,eg)) return true;
        if(seq2(es,hs,er,hr,eh)) return true;
        if(seq2(fs,gs,fr,gr,fg)) return true;
        if(seq2(fs,hs,fr,hr,fh)) return true;
        if(seq2(gs,hs,gr,hr,gh)) return true;

        if(convex && (DIM == 1)) return false;

        if(seq3(as,bs,cs,ar,br,cr,ab,ac,bc)) return true;
        if(seq3(as,bs,ds,ar,br,dr,ab,ad,bd)) return true;
        if(seq3(as,bs,es,ar,br,er,ab,ae,be)) return true;
        if(seq3(as,bs,fs,ar,br,fr,ab,af,bf)) return true;
        if(seq3(as,bs,gs,ar,br,gr,ab,ag,bg)) return true;
        if(seq3(as,bs,hs,ar,br,hr,ab,ah,bh)) return true;
        if(seq3(as,cs,ds,ar,cr,dr,ac,ad,cd)) return true;
        if(seq3(as,cs,es,ar,cr,er,ac,ae,ce)) return true;
        if(seq3(as,cs,fs,ar,cr,fr,ac,af,cf)) return true;
        if(seq3(as,cs,gs,ar,cr,gr,ac,ag,cg)) return true;
        if(seq3(as,cs,hs,ar,cr,hr,ac,ah,ch)) return true;
        if(seq3(as,ds,es,ar,dr,er,ad,ae,de)) return true;
        if(seq3(as,ds,fs,ar,dr,fr,ad,af,df)) return true;
        if(seq3(as,ds,gs,ar,dr,gr,ad,ag,dg)) return true;
        if(seq3(as,ds,hs,ar,dr,hr,ad,ah,dh)) return true;
        if(seq3(as,es,fs,ar,er,fr,ae,af,ef)) return true;
        if(seq3(as,es,gs,ar,er,gr,ae,ag,eg)) return true;
        if(seq3(as,es,hs,ar,er,hr,ae,ah,eh)) return true;
        if(seq3(as,fs,gs,ar,fr,gr,af,ag,fg)) return true;
        if(seq3(as,fs,hs,ar,fr,hr,af,ah,fh)) return true;
        if(seq3(as,gs,hs,ar,gr,hr,ag,ah,gh)) return true;
        if(seq3(bs,cs,ds,br,cr,dr,bc,bd,cd)) return true;
        if(seq3(bs,cs,es,br,cr,er,bc,be,ce)) return true;
        if(seq3(bs,cs,fs,br,cr,fr,bc,bf,cf)) return true;
        if(seq3(bs,cs,gs,br,cr,gr,bc,bg,cg)) return true;
        if(seq3(bs,cs,hs,br,cr,hr,bc,bh,ch)) return true;
        if(seq3(bs,ds,es,br,dr,er,bd,be,de)) return true;
        if(seq3(bs,ds,fs,br,dr,fr,bd,bf,df)) return true;
        if(seq3(bs,ds,gs,br,dr,gr,bd,bg,dg)) return true;
        if(seq3(bs,ds,hs,br,dr,hr,bd,bh,dh)) return true;
        if(seq3(bs,es,fs,br,er,fr,be,bf,ef)) return true;
        if(seq3(bs,es,gs,br,er,gr,be,bg,eg)) return true;
        if(seq3(bs,es,hs,br,er,hr,be,bh,eh)) return true;
        if(seq3(bs,fs,gs,br,fr,gr,bf,bg,fg)) return true;
        if(seq3(bs,fs,hs,br,fr,hr,bf,bh,fh)) return true;
        if(seq3(bs,gs,hs,br,gr,hr,bg,bh,gh)) return true;
        if(seq3(cs,ds,es,cr,dr,er,cd,ce,de)) return true;
        if(seq3(cs,ds,fs,cr,dr,fr,cd,cf,df)) return true;
        if(seq3(cs,ds,gs,cr,dr,gr,cd,cg,dg)) return true;
        if(seq3(cs,ds,hs,cr,dr,hr,cd,ch,dh)) return true;
        if(seq3(cs,es,fs,cr,er,fr,ce,cf,ef)) return true;
        if(seq3(cs,es,gs,cr,er,gr,ce,cg,eg)) return true;
        if(seq3(cs,es,hs,cr,er,hr,ce,ch,eh)) return true;
        if(seq3(cs,fs,gs,cr,fr,gr,cf,cg,fg)) return true;
        if(seq3(cs,fs,hs,cr,fr,hr,cf,ch,fh)) return true;
        if(seq3(cs,gs,hs,cr,gr,hr,cg,ch,gh)) return true;
        if(seq3(ds,es,fs,dr,er,fr,de,df,ef)) return true;
        if(seq3(ds,es,gs,dr,er,gr,de,dg,eg)) return true;
        if(seq3(ds,es,hs,dr,er,hr,de,dh,eh)) return true;
        if(seq3(ds,fs,gs,dr,fr,gr,df,dg,fg)) return true;
        if(seq3(ds,fs,hs,dr,fr,hr,df,dh,fh)) return true;
        if(seq3(ds,gs,hs,dr,gr,hr,dg,dh,gh)) return true;
        if(seq3(es,fs,gs,er,fr,gr,ef,eg,fg)) return true;
        if(seq3(es,fs,hs,er,fr,hr,ef,eh,fh)) return true;
        if(seq3(es,gs,hs,er,gr,hr,eg,eh,gh)) return true;
        if(seq3(fs,gs,hs,fr,gr,hr,fg,fh,gh)) return true;

        if(convex && (DIM == 2)) return false;

        if(seq4(as,bs,cs,ds,ar,br,cr,dr,ab,ac,ad,bc,bd,cd)) return true;
        if(seq4(as,bs,cs,es,ar,br,cr,er,ab,ac,ae,bc,be,ce)) return true;
        if(seq4(as,bs,cs,fs,ar,br,cr,fr,ab,ac,af,bc,bf,cf)) return true;
        if(seq4(as,bs,cs,gs,ar,br,cr,gr,ab,ac,ag,bc,bg,cg)) return true;
        if(seq4(as,bs,cs,hs,ar,br,cr,hr,ab,ac,ah,bc,bh,ch)) return true;
        if(seq4(as,bs,ds,es,ar,br,dr,er,ab,ad,ae,bd,be,de)) return true;
        if(seq4(as,bs,ds,fs,ar,br,dr,fr,ab,ad,af,bd,bf,df)) return true;
        if(seq4(as,bs,ds,gs,ar,br,dr,gr,ab,ad,ag,bd,bg,dg)) return true;
        if(seq4(as,bs,ds,hs,ar,br,dr,hr,ab,ad,ah,bd,bh,dh)) return true;
        if(seq4(as,bs,es,fs,ar,br,er,fr,ab,ae,af,be,bf,ef)) return true;
        if(seq4(as,bs,es,gs,ar,br,er,gr,ab,ae,ag,be,bg,eg)) return true;
        if(seq4(as,bs,es,hs,ar,br,er,hr,ab,ae,ah,be,bh,eh)) return true;
        if(seq4(as,bs,fs,gs,ar,br,fr,gr,ab,af,ag,bf,bg,fg)) return true;
        if(seq4(as,bs,fs,hs,ar,br,fr,hr,ab,af,ah,bf,bh,fh)) return true;
        if(seq4(as,bs,gs,hs,ar,br,gr,hr,ab,ag,ah,bg,bh,gh)) return true;
        if(seq4(as,cs,ds,es,ar,cr,dr,er,ac,ad,ae,cd,ce,de)) return true;
        if(seq4(as,cs,ds,fs,ar,cr,dr,fr,ac,ad,af,cd,cf,df)) return true;
        if(seq4(as,cs,ds,gs,ar,cr,dr,gr,ac,ad,ag,cd,cg,dg)) return true;
        if(seq4(as,cs,ds,hs,ar,cr,dr,hr,ac,ad,ah,cd,ch,dh)) return true;
        if(seq4(as,cs,es,fs,ar,cr,er,fr,ac,ae,af,ce,cf,ef)) return true;
        if(seq4(as,cs,es,gs,ar,cr,er,gr,ac,ae,ag,ce,cg,eg)) return true;
        if(seq4(as,cs,es,hs,ar,cr,er,hr,ac,ae,ah,ce,ch,eh)) return true;
        if(seq4(as,cs,fs,gs,ar,cr,fr,gr,ac,af,ag,cf,cg,fg)) return true;
        if(seq4(as,cs,fs,hs,ar,cr,fr,hr,ac,af,ah,cf,ch,fh)) return true;
        if(seq4(as,cs,gs,hs,ar,cr,gr,hr,ac,ag,ah,cg,ch,gh)) return true;
        if(seq4(as,ds,es,fs,ar,dr,er,fr,ad,ae,af,de,df,ef)) return true;
        if(seq4(as,ds,es,gs,ar,dr,er,gr,ad,ae,ag,de,dg,eg)) return true;
        if(seq4(as,ds,es,hs,ar,dr,er,hr,ad,ae,ah,de,dh,eh)) return true;
        if(seq4(as,ds,fs,gs,ar,dr,fr,gr,ad,af,ag,df,dg,fg)) return true;
        if(seq4(as,ds,fs,hs,ar,dr,fr,hr,ad,af,ah,df,dh,fh)) return true;
        if(seq4(as,ds,gs,hs,ar,dr,gr,hr,ad,ag,ah,dg,dh,gh)) return true;
        if(seq4(as,es,fs,gs,ar,er,fr,gr,ae,af,ag,ef,eg,fg)) return true;
        if(seq4(as,es,fs,hs,ar,er,fr,hr,ae,af,ah,ef,eh,fh)) return true;
        if(seq4(as,es,gs,hs,ar,er,gr,hr,ae,ag,ah,eg,eh,gh)) return true;
        if(seq4(as,fs,gs,hs,ar,fr,gr,hr,af,ag,ah,fg,fh,gh)) return true;
        if(seq4(bs,cs,ds,es,br,cr,dr,er,bc,bd,be,cd,ce,de)) return true;
        if(seq4(bs,cs,ds,fs,br,cr,dr,fr,bc,bd,bf,cd,cf,df)) return true;
        if(seq4(bs,cs,ds,gs,br,cr,dr,gr,bc,bd,bg,cd,cg,dg)) return true;
        if(seq4(bs,cs,ds,hs,br,cr,dr,hr,bc,bd,bh,cd,ch,dh)) return true;
        if(seq4(bs,cs,es,fs,br,cr,er,fr,bc,be,bf,ce,cf,ef)) return true;
        if(seq4(bs,cs,es,gs,br,cr,er,gr,bc,be,bg,ce,cg,eg)) return true;
        if(seq4(bs,cs,es,hs,br,cr,er,hr,bc,be,bh,ce,ch,eh)) return true;
        if(seq4(bs,cs,fs,gs,br,cr,fr,gr,bc,bf,bg,cf,cg,fg)) return true;
        if(seq4(bs,cs,fs,hs,br,cr,fr,hr,bc,bf,bh,cf,ch,fh)) return true;
        if(seq4(bs,cs,gs,hs,br,cr,gr,hr,bc,bg,bh,cg,ch,gh)) return true;
        if(seq4(bs,ds,es,fs,br,dr,er,fr,bd,be,bf,de,df,ef)) return true;
        if(seq4(bs,ds,es,gs,br,dr,er,gr,bd,be,bg,de,dg,eg)) return true;
        if(seq4(bs,ds,es,hs,br,dr,er,hr,bd,be,bh,de,dh,eh)) return true;
        if(seq4(bs,ds,fs,gs,br,dr,fr,gr,bd,bf,bg,df,dg,fg)) return true;
        if(seq4(bs,ds,fs,hs,br,dr,fr,hr,bd,bf,bh,df,dh,fh)) return true;
        if(seq4(bs,ds,gs,hs,br,dr,gr,hr,bd,bg,bh,dg,dh,gh)) return true;
        if(seq4(bs,es,fs,gs,br,er,fr,gr,be,bf,bg,ef,eg,fg)) return true;
        if(seq4(bs,es,fs,hs,br,er,fr,hr,be,bf,bh,ef,eh,fh)) return true;
        if(seq4(bs,es,gs,hs,br,er,gr,hr,be,bg,bh,eg,eh,gh)) return true;
        if(seq4(bs,fs,gs,hs,br,fr,gr,hr,bf,bg,bh,fg,fh,gh)) return true;
        if(seq4(cs,ds,es,fs,cr,dr,er,fr,cd,ce,cf,de,df,ef)) return true;
        if(seq4(cs,ds,es,gs,cr,dr,er,gr,cd,ce,cg,de,dg,eg)) return true;
        if(seq4(cs,ds,es,hs,cr,dr,er,hr,cd,ce,ch,de,dh,eh)) return true;
        if(seq4(cs,ds,fs,gs,cr,dr,fr,gr,cd,cf,cg,df,dg,fg)) return true;
        if(seq4(cs,ds,fs,hs,cr,dr,fr,hr,cd,cf,ch,df,dh,fh)) return true;
        if(seq4(cs,ds,gs,hs,cr,dr,gr,hr,cd,cg,ch,dg,dh,gh)) return true;
        if(seq4(cs,es,fs,gs,cr,er,fr,gr,ce,cf,cg,ef,eg,fg)) return true;
        if(seq4(cs,es,fs,hs,cr,er,fr,hr,ce,cf,ch,ef,eh,fh)) return true;
        if(seq4(cs,es,gs,hs,cr,er,gr,hr,ce,cg,ch,eg,eh,gh)) return true;
        if(seq4(cs,fs,gs,hs,cr,fr,gr,hr,cf,cg,ch,fg,fh,gh)) return true;
        if(seq4(ds,es,fs,gs,dr,er,fr,gr,de,df,dg,ef,eg,fg)) return true;
        if(seq4(ds,es,fs,hs,dr,er,fr,hr,de,df,dh,ef,eh,fh)) return true;
        if(seq4(ds,es,gs,hs,dr,er,gr,hr,de,dg,dh,eg,eh,gh)) return true;
        if(seq4(ds,fs,gs,hs,dr,fr,gr,hr,df,dg,dh,fg,fh,gh)) return true;
        if(seq4(es,fs,gs,hs,er,fr,gr,hr,ef,eg,eh,fg,fh,gh)) return true;

        if(convex && (DIM == 3)) return false;

        if(seq5(as,bs,cs,ds,es,ar,br,cr,dr,er,ab,ac,ad,ae,bc,bd,be,cd,ce,de)) return true;
        if(seq5(as,bs,cs,ds,fs,ar,br,cr,dr,fr,ab,ac,ad,af,bc,bd,bf,cd,cf,df)) return true;
        if(seq5(as,bs,cs,ds,gs,ar,br,cr,dr,gr,ab,ac,ad,ag,bc,bd,bg,cd,cg,dg)) return true;
        if(seq5(as,bs,cs,ds,hs,ar,br,cr,dr,hr,ab,ac,ad,ah,bc,bd,bh,cd,ch,dh)) return true;
        if(seq5(as,bs,cs,es,fs,ar,br,cr,er,fr,ab,ac,ae,af,bc,be,bf,ce,cf,ef)) return true;
        if(seq5(as,bs,cs,es,gs,ar,br,cr,er,gr,ab,ac,ae,ag,bc,be,bg,ce,cg,eg)) return true;
        if(seq5(as,bs,cs,es,hs,ar,br,cr,er,hr,ab,ac,ae,ah,bc,be,bh,ce,ch,eh)) return true;
        if(seq5(as,bs,cs,fs,gs,ar,br,cr,fr,gr,ab,ac,af,ag,bc,bf,bg,cf,cg,fg)) return true;
        if(seq5(as,bs,cs,fs,hs,ar,br,cr,fr,hr,ab,ac,af,ah,bc,bf,bh,cf,ch,fh)) return true;
        if(seq5(as,bs,cs,gs,hs,ar,br,cr,gr,hr,ab,ac,ag,ah,bc,bg,bh,cg,ch,gh)) return true;
        if(seq5(as,bs,ds,es,fs,ar,br,dr,er,fr,ab,ad,ae,af,bd,be,bf,de,df,ef)) return true;
        if(seq5(as,bs,ds,es,gs,ar,br,dr,er,gr,ab,ad,ae,ag,bd,be,bg,de,dg,eg)) return true;
        if(seq5(as,bs,ds,es,hs,ar,br,dr,er,hr,ab,ad,ae,ah,bd,be,bh,de,dh,eh)) return true;
        if(seq5(as,bs,ds,fs,gs,ar,br,dr,fr,gr,ab,ad,af,ag,bd,bf,bg,df,dg,fg)) return true;
        if(seq5(as,bs,ds,fs,hs,ar,br,dr,fr,hr,ab,ad,af,ah,bd,bf,bh,df,dh,fh)) return true;
        if(seq5(as,bs,ds,gs,hs,ar,br,dr,gr,hr,ab,ad,ag,ah,bd,bg,bh,dg,dh,gh)) return true;
        if(seq5(as,bs,es,fs,gs,ar,br,er,fr,gr,ab,ae,af,ag,be,bf,bg,ef,eg,fg)) return true;
        if(seq5(as,bs,es,fs,hs,ar,br,er,fr,hr,ab,ae,af,ah,be,bf,bh,ef,eh,fh)) return true;
        if(seq5(as,bs,es,gs,hs,ar,br,er,gr,hr,ab,ae,ag,ah,be,bg,bh,eg,eh,gh)) return true;
        if(seq5(as,bs,fs,gs,hs,ar,br,fr,gr,hr,ab,af,ag,ah,bf,bg,bh,fg,fh,gh)) return true;
        if(seq5(as,cs,ds,es,fs,ar,cr,dr,er,fr,ac,ad,ae,af,cd,ce,cf,de,df,ef)) return true;
        if(seq5(as,cs,ds,es,gs,ar,cr,dr,er,gr,ac,ad,ae,ag,cd,ce,cg,de,dg,eg)) return true;
        if(seq5(as,cs,ds,es,hs,ar,cr,dr,er,hr,ac,ad,ae,ah,cd,ce,ch,de,dh,eh)) return true;
        if(seq5(as,cs,ds,fs,gs,ar,cr,dr,fr,gr,ac,ad,af,ag,cd,cf,cg,df,dg,fg)) return true;
        if(seq5(as,cs,ds,fs,hs,ar,cr,dr,fr,hr,ac,ad,af,ah,cd,cf,ch,df,dh,fh)) return true;
        if(seq5(as,cs,ds,gs,hs,ar,cr,dr,gr,hr,ac,ad,ag,ah,cd,cg,ch,dg,dh,gh)) return true;
        if(seq5(as,cs,es,fs,gs,ar,cr,er,fr,gr,ac,ae,af,ag,ce,cf,cg,ef,eg,fg)) return true;
        if(seq5(as,cs,es,fs,hs,ar,cr,er,fr,hr,ac,ae,af,ah,ce,cf,ch,ef,eh,fh)) return true;
        if(seq5(as,cs,es,gs,hs,ar,cr,er,gr,hr,ac,ae,ag,ah,ce,cg,ch,eg,eh,gh)) return true;
        if(seq5(as,cs,fs,gs,hs,ar,cr,fr,gr,hr,ac,af,ag,ah,cf,cg,ch,fg,fh,gh)) return true;
        if(seq5(as,ds,es,fs,gs,ar,dr,er,fr,gr,ad,ae,af,ag,de,df,dg,ef,eg,fg)) return true;
        if(seq5(as,ds,es,fs,hs,ar,dr,er,fr,hr,ad,ae,af,ah,de,df,dh,ef,eh,fh)) return true;
        if(seq5(as,ds,es,gs,hs,ar,dr,er,gr,hr,ad,ae,ag,ah,de,dg,dh,eg,eh,gh)) return true;
        if(seq5(as,ds,fs,gs,hs,ar,dr,fr,gr,hr,ad,af,ag,ah,df,dg,dh,fg,fh,gh)) return true;
        if(seq5(as,es,fs,gs,hs,ar,er,fr,gr,hr,ae,af,ag,ah,ef,eg,eh,fg,fh,gh)) return true;
        if(seq5(bs,cs,ds,es,fs,br,cr,dr,er,fr,bc,bd,be,bf,cd,ce,cf,de,df,ef)) return true;
        if(seq5(bs,cs,ds,es,gs,br,cr,dr,er,gr,bc,bd,be,bg,cd,ce,cg,de,dg,eg)) return true;
        if(seq5(bs,cs,ds,es,hs,br,cr,dr,er,hr,bc,bd,be,bh,cd,ce,ch,de,dh,eh)) return true;
        if(seq5(bs,cs,ds,fs,gs,br,cr,dr,fr,gr,bc,bd,bf,bg,cd,cf,cg,df,dg,fg)) return true;
        if(seq5(bs,cs,ds,fs,hs,br,cr,dr,fr,hr,bc,bd,bf,bh,cd,cf,ch,df,dh,fh)) return true;
        if(seq5(bs,cs,ds,gs,hs,br,cr,dr,gr,hr,bc,bd,bg,bh,cd,cg,ch,dg,dh,gh)) return true;
        if(seq5(bs,cs,es,fs,gs,br,cr,er,fr,gr,bc,be,bf,bg,ce,cf,cg,ef,eg,fg)) return true;
        if(seq5(bs,cs,es,fs,hs,br,cr,er,fr,hr,bc,be,bf,bh,ce,cf,ch,ef,eh,fh)) return true;
        if(seq5(bs,cs,es,gs,hs,br,cr,er,gr,hr,bc,be,bg,bh,ce,cg,ch,eg,eh,gh)) return true;
        if(seq5(bs,cs,fs,gs,hs,br,cr,fr,gr,hr,bc,bf,bg,bh,cf,cg,ch,fg,fh,gh)) return true;
        if(seq5(bs,ds,es,fs,gs,br,dr,er,fr,gr,bd,be,bf,bg,de,df,dg,ef,eg,fg)) return true;
        if(seq5(bs,ds,es,fs,hs,br,dr,er,fr,hr,bd,be,bf,bh,de,df,dh,ef,eh,fh)) return true;
        if(seq5(bs,ds,es,gs,hs,br,dr,er,gr,hr,bd,be,bg,bh,de,dg,dh,eg,eh,gh)) return true;
        if(seq5(bs,ds,fs,gs,hs,br,dr,fr,gr,hr,bd,bf,bg,bh,df,dg,dh,fg,fh,gh)) return true;
        if(seq5(bs,es,fs,gs,hs,br,er,fr,gr,hr,be,bf,bg,bh,ef,eg,eh,fg,fh,gh)) return true;
        if(seq5(cs,ds,es,fs,gs,cr,dr,er,fr,gr,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;
        if(seq5(cs,ds,es,fs,hs,cr,dr,er,fr,hr,cd,ce,cf,ch,de,df,dh,ef,eh,fh)) return true;
        if(seq5(cs,ds,es,gs,hs,cr,dr,er,gr,hr,cd,ce,cg,ch,de,dg,dh,eg,eh,gh)) return true;
        if(seq5(cs,ds,fs,gs,hs,cr,dr,fr,gr,hr,cd,cf,cg,ch,df,dg,dh,fg,fh,gh)) return true;
        if(seq5(cs,es,fs,gs,hs,cr,er,fr,gr,hr,ce,cf,cg,ch,ef,eg,eh,fg,fh,gh)) return true;
        if(seq5(ds,es,fs,gs,hs,dr,er,fr,gr,hr,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;

        if(convex && (DIM == 4)) return false;

        if(seq6(as,bs,cs,ds,es,fs,ar,br,cr,dr,er,fr,ab,ac,ad,ae,af,bc,bd,be,bf,cd,ce,cf,de,df,ef)) return true;
        if(seq6(as,bs,cs,ds,es,gs,ar,br,cr,dr,er,gr,ab,ac,ad,ae,ag,bc,bd,be,bg,cd,ce,cg,de,dg,eg)) return true;
        if(seq6(as,bs,cs,ds,es,hs,ar,br,cr,dr,er,hr,ab,ac,ad,ae,ah,bc,bd,be,bh,cd,ce,ch,de,dh,eh)) return true;
        if(seq6(as,bs,cs,ds,fs,gs,ar,br,cr,dr,fr,gr,ab,ac,ad,af,ag,bc,bd,bf,bg,cd,cf,cg,df,dg,fg)) return true;
        if(seq6(as,bs,cs,ds,fs,hs,ar,br,cr,dr,fr,hr,ab,ac,ad,af,ah,bc,bd,bf,bh,cd,cf,ch,df,dh,fh)) return true;
        if(seq6(as,bs,cs,ds,gs,hs,ar,br,cr,dr,gr,hr,ab,ac,ad,ag,ah,bc,bd,bg,bh,cd,cg,ch,dg,dh,gh)) return true;
        if(seq6(as,bs,cs,es,fs,gs,ar,br,cr,er,fr,gr,ab,ac,ae,af,ag,bc,be,bf,bg,ce,cf,cg,ef,eg,fg)) return true;
        if(seq6(as,bs,cs,es,fs,hs,ar,br,cr,er,fr,hr,ab,ac,ae,af,ah,bc,be,bf,bh,ce,cf,ch,ef,eh,fh)) return true;
        if(seq6(as,bs,cs,es,gs,hs,ar,br,cr,er,gr,hr,ab,ac,ae,ag,ah,bc,be,bg,bh,ce,cg,ch,eg,eh,gh)) return true;
        if(seq6(as,bs,cs,fs,gs,hs,ar,br,cr,fr,gr,hr,ab,ac,af,ag,ah,bc,bf,bg,bh,cf,cg,ch,fg,fh,gh)) return true;
        if(seq6(as,bs,ds,es,fs,gs,ar,br,dr,er,fr,gr,ab,ad,ae,af,ag,bd,be,bf,bg,de,df,dg,ef,eg,fg)) return true;
        if(seq6(as,bs,ds,es,fs,hs,ar,br,dr,er,fr,hr,ab,ad,ae,af,ah,bd,be,bf,bh,de,df,dh,ef,eh,fh)) return true;
        if(seq6(as,bs,ds,es,gs,hs,ar,br,dr,er,gr,hr,ab,ad,ae,ag,ah,bd,be,bg,bh,de,dg,dh,eg,eh,gh)) return true;
        if(seq6(as,bs,ds,fs,gs,hs,ar,br,dr,fr,gr,hr,ab,ad,af,ag,ah,bd,bf,bg,bh,df,dg,dh,fg,fh,gh)) return true;
        if(seq6(as,bs,es,fs,gs,hs,ar,br,er,fr,gr,hr,ab,ae,af,ag,ah,be,bf,bg,bh,ef,eg,eh,fg,fh,gh)) return true;
        if(seq6(as,cs,ds,es,fs,gs,ar,cr,dr,er,fr,gr,ac,ad,ae,af,ag,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;
        if(seq6(as,cs,ds,es,fs,hs,ar,cr,dr,er,fr,hr,ac,ad,ae,af,ah,cd,ce,cf,ch,de,df,dh,ef,eh,fh)) return true;
        if(seq6(as,cs,ds,es,gs,hs,ar,cr,dr,er,gr,hr,ac,ad,ae,ag,ah,cd,ce,cg,ch,de,dg,dh,eg,eh,gh)) return true;
        if(seq6(as,cs,ds,fs,gs,hs,ar,cr,dr,fr,gr,hr,ac,ad,af,ag,ah,cd,cf,cg,ch,df,dg,dh,fg,fh,gh)) return true;
        if(seq6(as,cs,es,fs,gs,hs,ar,cr,er,fr,gr,hr,ac,ae,af,ag,ah,ce,cf,cg,ch,ef,eg,eh,fg,fh,gh)) return true;
        if(seq6(as,ds,es,fs,gs,hs,ar,dr,er,fr,gr,hr,ad,ae,af,ag,ah,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;
        if(seq6(bs,cs,ds,es,fs,gs,br,cr,dr,er,fr,gr,bc,bd,be,bf,bg,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;
        if(seq6(bs,cs,ds,es,fs,hs,br,cr,dr,er,fr,hr,bc,bd,be,bf,bh,cd,ce,cf,ch,de,df,dh,ef,eh,fh)) return true;
        if(seq6(bs,cs,ds,es,gs,hs,br,cr,dr,er,gr,hr,bc,bd,be,bg,bh,cd,ce,cg,ch,de,dg,dh,eg,eh,gh)) return true;
        if(seq6(bs,cs,ds,fs,gs,hs,br,cr,dr,fr,gr,hr,bc,bd,bf,bg,bh,cd,cf,cg,ch,df,dg,dh,fg,fh,gh)) return true;
        if(seq6(bs,cs,es,fs,gs,hs,br,cr,er,fr,gr,hr,bc,be,bf,bg,bh,ce,cf,cg,ch,ef,eg,eh,fg,fh,gh)) return true;
        if(seq6(bs,ds,es,fs,gs,hs,br,dr,er,fr,gr,hr,bd,be,bf,bg,bh,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;
        if(seq6(cs,ds,es,fs,gs,hs,cr,dr,er,fr,gr,hr,cd,ce,cf,cg,ch,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;

        if(convex && (DIM == 5)) return false;

        //if(seq7(as,bs,cs,ds,es,fs,gs,ar,br,cr,dr,er,fr,gr,ab,ac,ad,ae,af,ag,bc,bd,be,bf,bg,cd,ce,cf,cg,de,df,dg,ef,eg,fg)) return true;
        //if(seq7(as,bs,cs,ds,es,fs,hs,ar,br,cr,dr,er,fr,hr,ab,ac,ad,ae,af,ah,bc,bd,be,bf,bh,cd,ce,cf,ch,de,df,dh,ef,eh,fh)) return true;
        //if(seq7(as,bs,cs,ds,es,gs,hs,ar,br,cr,dr,er,gr,hr,ab,ac,ad,ae,ag,ah,bc,bd,be,bg,bh,cd,ce,cg,ch,de,dg,dh,eg,eh,gh)) return true;
        //if(seq7(as,bs,cs,ds,fs,gs,hs,ar,br,cr,dr,fr,gr,hr,ab,ac,ad,af,ag,ah,bc,bd,bf,bg,bh,cd,cf,cg,ch,df,dg,dh,fg,fh,gh)) return true;
        //if(seq7(as,bs,cs,es,fs,gs,hs,ar,br,cr,er,fr,gr,hr,ab,ac,ae,af,ag,ah,bc,be,bf,bg,bh,ce,cf,cg,ch,ef,eg,eh,fg,fh,gh)) return true;
        //if(seq7(as,bs,ds,es,fs,gs,hs,ar,br,dr,er,fr,gr,hr,ab,ad,ae,af,ag,ah,bd,be,bf,bg,bh,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;
        //if(seq7(as,cs,ds,es,fs,gs,hs,ar,cr,dr,er,fr,gr,hr,ac,ad,ae,af,ag,ah,cd,ce,cf,cg,ch,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;
        //if(seq7(bs,cs,ds,es,fs,gs,hs,br,cr,dr,er,fr,gr,hr,bc,bd,be,bf,bg,bh,cd,ce,cf,cg,ch,de,df,dg,dh,ef,eg,eh,fg,fh,gh)) return true;

        if(convex && (DIM == 6)) return false;

        //return seq8(as,bs,cs,ds,es,fs,gs,hs,ar,br,cr,dr,er,fr,gr,hr,ab,ac,ad,ae,af,ag,ah,bc,bd,be,bf,bg,bh,cd,ce,cf,cg,ch,de,df,dg,dh,ef,eg,eh,fg,fh,gh);
        return false;
    }


/*  OverlapReal op3(OverlapReal ab,OverlapReal ac,OverlapReal ad,
               OverlapReal bc,OverlapReal bd,
               OverlapReal cd)
    {
        return 2*(ab*cd*(ac+bd+ad+bc)+ad*bc*(ab+cd+ac+bd)+ac*bd*(ab+cd+ad+bc));
    }

    OverlapReal op4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
               OverlapReal bc,OverlapReal bd,OverlapReal be,
               OverlapReal cd,OverlapReal ce,
               OverlapReal de)
    {
        return 2*(ac*ae*bc*bd+ad*ae*bc*bd+ac*ad*bc*be+ad*ae*bc*be+ac*ad*bd*be+ac*ae*bd*be+
                  ab*ae*bc*cd+ad*ae*bc*cd+ab*ae*bd*cd+ac*ae*bd*cd+ab*ac*be*cd+ab*ad*be*cd+
                  ac*ae*be*cd+ad*ae*be*cd+ad*bc*be*cd+ae*bc*be*cd+ac*bd*be*cd+ae*bd*be*cd+
                  ab*ad*bc*ce+ad*ae*bc*ce+ab*ac*bd*ce+ac*ad*bd*ce+ab*ae*bd*ce+ad*ae*bd*ce+
                  ad*bc*bd*ce+ae*bc*bd*ce+ab*ad*be*ce+ac*ad*be*ce+ac*bd*be*ce+ad*bd*be*ce+
                  ab*ad*cd*ce+ab*ae*cd*ce+ab*bd*cd*ce+ae*bd*cd*ce+ab*be*cd*ce+ad*be*cd*ce+
                  ab*ad*bc*de+ac*ad*bc*de+ab*ae*bc*de+ac*ae*bc*de+ab*ac*bd*de+ac*ae*bd*de+
                  ac*bc*bd*de+ae*bc*bd*de+ab*ac*be*de+ac*ad*be*de+ac*bc*be*de+ad*bc*be*de+
                  ab*ac*cd*de+ab*ae*cd*de+ab*bc*cd*de+ae*bc*cd*de+ab*be*cd*de+ac*be*cd*de+
                  ab*ac*ce*de+ab*ad*ce*de+ab*bc*ce*de+ad*bc*ce*de+ab*bd*ce*de+ac*bd*ce*de);

    }

    OverlapReal op5(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
               OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
               OverlapReal cd,OverlapReal ce,OverlapReal cf,
               OverlapReal de,OverlapReal df,
               OverlapReal ef)
    {
        return 2*(ad*af*bc*be*cd+ae*af*bc*be*cd+ac*af*bd*be*cd+ae*af*bd*be*cd+ad*ae*bc*bf*cd+ae*af*bc*bf*cd+
                  ac*ae*bd*bf*cd+ae*af*bd*bf*cd+ac*ae*be*bf*cd+ad*ae*be*bf*cd+ac*af*be*bf*cd+ad*af*be*bf*cd+
                  ad*af*bc*bd*ce+ae*af*bc*bd*ce+ac*af*bd*be*ce+ad*af*bd*be*ce+ad*ae*bc*bf*ce+ad*af*bc*bf*ce+
                  ac*ad*bd*bf*ce+ad*ae*bd*bf*ce+ac*af*bd*bf*ce+ae*af*bd*bf*ce+ac*ad*be*bf*ce+ad*af*be*bf*ce+
                  ab*af*bd*cd*ce+ae*af*bd*cd*ce+ab*af*be*cd*ce+ad*af*be*cd*ce+ab*ad*bf*cd*ce+ab*ae*bf*cd*ce+
                  ad*af*bf*cd*ce+ae*af*bf*cd*ce+ae*bd*bf*cd*ce+af*bd*bf*cd*ce+ad*be*bf*cd*ce+af*be*bf*cd*ce+
                  ad*ae*bc*bd*cf+ae*af*bc*bd*cf+ad*ae*bc*be*cf+ad*af*bc*be*cf+ac*ad*bd*be*cf+ac*ae*bd*be*cf+
                  ad*af*bd*be*cf+ae*af*bd*be*cf+ac*ae*bd*bf*cf+ad*ae*bd*bf*cf+ac*ad*be*bf*cf+ad*ae*be*bf*cf+
                  ab*ae*bd*cd*cf+ae*af*bd*cd*cf+ab*ad*be*cd*cf+ad*ae*be*cd*cf+ab*af*be*cd*cf+ae*af*be*cd*cf+
                  ae*bd*be*cd*cf+af*bd*be*cd*cf+ab*ae*bf*cd*cf+ad*ae*bf*cd*cf+ad*be*bf*cd*cf+ae*be*bf*cd*cf+
                  ab*ae*bd*ce*cf+ad*ae*bd*ce*cf+ab*af*bd*ce*cf+ad*af*bd*ce*cf+ab*ad*be*ce*cf+ad*af*be*ce*cf+
                  ad*bd*be*ce*cf+af*bd*be*ce*cf+ab*ad*bf*ce*cf+ad*ae*bf*ce*cf+ad*bd*bf*ce*cf+ae*bd*bf*ce*cf+
                  ac*af*bc*bd*de+ae*af*bc*bd*de+ac*af*bc*be*de+ad*af*bc*be*de+ac*ad*bc*bf*de+ac*ae*bc*bf*de+
                  ad*af*bc*bf*de+ae*af*bc*bf*de+ac*ae*bd*bf*de+ac*af*bd*bf*de+ac*ad*be*bf*de+ac*af*be*bf*de+
                  ab*af*bc*cd*de+ae*af*bc*cd*de+ab*af*be*cd*de+ac*af*be*cd*de+ab*ac*bf*cd*de+ab*ae*bf*cd*de+
                  ac*af*bf*cd*de+ae*af*bf*cd*de+ae*bc*bf*cd*de+af*bc*bf*cd*de+ac*be*bf*cd*de+af*be*bf*cd*de+
                  ab*af*bc*ce*de+ad*af*bc*ce*de+ab*af*bd*ce*de+ac*af*bd*ce*de+ab*ac*bf*ce*de+ab*ad*bf*ce*de+
                  ac*af*bf*ce*de+ad*af*bf*ce*de+ad*bc*bf*ce*de+af*bc*bf*ce*de+ac*bd*bf*ce*de+af*bd*bf*ce*de+
                  ab*ad*bc*cf*de+ab*ae*bc*cf*de+ad*af*bc*cf*de+ae*af*bc*cf*de+ab*ac*bd*cf*de+ac*ae*bd*cf*de+
                  ab*af*bd*cf*de+ae*af*bd*cf*de+ae*bc*bd*cf*de+af*bc*bd*cf*de+ab*ac*be*cf*de+ac*ad*be*cf*de+
                  ab*af*be*cf*de+ad*af*be*cf*de+ad*bc*be*cf*de+af*bc*be*cf*de+ab*ad*bf*cf*de+ac*ad*bf*cf*de+
                  ab*ae*bf*cf*de+ac*ae*bf*cf*de+ac*bd*bf*cf*de+ae*bd*bf*cf*de+ac*be*bf*cf*de+ad*be*bf*cf*de+
                  ab*ae*cd*cf*de+ab*af*cd*cf*de+ab*be*cd*cf*de+af*be*cd*cf*de+ab*bf*cd*cf*de+ae*bf*cd*cf*de+
                  ab*ad*ce*cf*de+ab*af*ce*cf*de+ab*bd*ce*cf*de+af*bd*ce*cf*de+ab*bf*ce*cf*de+ad*bf*ce*cf*de+
                  ac*ae*bc*bd*df+ae*af*bc*bd*df+ac*ad*bc*be*df+ad*ae*bc*be*df+ac*af*bc*be*df+ae*af*bc*be*df+
                  ac*ae*bd*be*df+ac*af*bd*be*df+ac*ae*bc*bf*df+ad*ae*bc*bf*df+ac*ad*be*bf*df+ac*ae*be*bf*df+
                  ab*ae*bc*cd*df+ae*af*bc*cd*df+ab*ac*be*cd*df+ac*ae*be*cd*df+ab*af*be*cd*df+ae*af*be*cd*df+
                  ae*bc*be*cd*df+af*bc*be*cd*df+ab*ae*bf*cd*df+ac*ae*bf*cd*df+ac*be*bf*cd*df+ae*be*bf*cd*df+
                  ab*ad*bc*ce*df+ad*ae*bc*ce*df+ab*af*bc*ce*df+ae*af*bc*ce*df+ab*ac*bd*ce*df+ab*ae*bd*ce*df+
                  ac*af*bd*ce*df+ae*af*bd*ce*df+ae*bc*bd*ce*df+af*bc*bd*ce*df+ab*ad*be*ce*df+ac*ad*be*ce*df+
                  ab*af*be*ce*df+ac*af*be*ce*df+ac*bd*be*ce*df+af*bd*be*ce*df+ab*ac*bf*ce*df+ac*ad*bf*ce*df+
                  ab*ae*bf*ce*df+ad*ae*bf*ce*df+ad*bc*bf*ce*df+ae*bc*bf*ce*df+ac*be*bf*ce*df+ad*be*bf*ce*df+
                  ab*ae*cd*ce*df+ab*af*cd*ce*df+ab*be*cd*ce*df+af*be*cd*ce*df+ab*bf*cd*ce*df+ae*bf*cd*ce*df+
                  ab*ae*bc*cf*df+ad*ae*bc*cf*df+ab*ae*bd*cf*df+ac*ae*bd*cf*df+ab*ac*be*cf*df+ab*ad*be*cf*df+
                  ac*ae*be*cf*df+ad*ae*be*cf*df+ad*bc*be*cf*df+ae*bc*be*cf*df+ac*bd*be*cf*df+ae*bd*be*cf*df+
                  ab*ad*ce*cf*df+ab*ae*ce*cf*df+ab*bd*ce*cf*df+ae*bd*ce*cf*df+ab*be*ce*cf*df+ad*be*ce*cf*df+
                  ab*ae*bc*de*df+ac*ae*bc*de*df+ab*af*bc*de*df+ac*af*bc*de*df+ab*ac*be*de*df+ac*af*be*de*df+
                  ac*bc*be*de*df+af*bc*be*de*df+ab*ac*bf*de*df+ac*ae*bf*de*df+ac*bc*bf*de*df+ae*bc*bf*de*df+
                  ab*ac*ce*de*df+ab*af*ce*de*df+ab*bc*ce*de*df+af*bc*ce*de*df+ab*bf*ce*de*df+ac*bf*ce*de*df+
                  ab*ac*cf*de*df+ab*ae*cf*de*df+ab*bc*cf*de*df+ae*bc*cf*de*df+ab*be*cf*de*df+ac*be*cf*de*df+
                  ac*ae*bc*bd*ef+ad*ae*bc*bd*ef+ac*af*bc*bd*ef+ad*af*bc*bd*ef+ac*ad*bc*be*ef+ad*af*bc*be*ef+
                  ac*ad*bd*be*ef+ac*af*bd*be*ef+ac*ad*bc*bf*ef+ad*ae*bc*bf*ef+ac*ad*bd*bf*ef+ac*ae*bd*bf*ef+
                  ab*ae*bc*cd*ef+ad*ae*bc*cd*ef+ab*af*bc*cd*ef+ad*af*bc*cd*ef+ab*ae*bd*cd*ef+ac*ae*bd*cd*ef+
                  ab*af*bd*cd*ef+ac*af*bd*cd*ef+ab*ac*be*cd*ef+ab*ad*be*cd*ef+ac*af*be*cd*ef+ad*af*be*cd*ef+
                  ad*bc*be*cd*ef+af*bc*be*cd*ef+ac*bd*be*cd*ef+af*bd*be*cd*ef+ab*ac*bf*cd*ef+ab*ad*bf*cd*ef+
                  ac*ae*bf*cd*ef+ad*ae*bf*cd*ef+ad*bc*bf*cd*ef+ae*bc*bf*cd*ef+ac*bd*bf*cd*ef+ae*bd*bf*cd*ef+
                  ab*ad*bc*ce*ef+ad*af*bc*ce*ef+ab*ac*bd*ce*ef+ac*ad*bd*ce*ef+ab*af*bd*ce*ef+ad*af*bd*ce*ef+
                  ad*bc*bd*ce*ef+af*bc*bd*ce*ef+ab*ad*bf*ce*ef+ac*ad*bf*ce*ef+ac*bd*bf*ce*ef+ad*bd*bf*ce*ef+
                  ab*ad*cd*ce*ef+ab*af*cd*ce*ef+ab*bd*cd*ce*ef+af*bd*cd*ce*ef+ab*bf*cd*ce*ef+ad*bf*cd*ce*ef+
                  ab*ad*bc*cf*ef+ad*ae*bc*cf*ef+ab*ac*bd*cf*ef+ac*ad*bd*cf*ef+ab*ae*bd*cf*ef+ad*ae*bd*cf*ef+
                  ad*bc*bd*cf*ef+ae*bc*bd*cf*ef+ab*ad*be*cf*ef+ac*ad*be*cf*ef+ac*bd*be*cf*ef+ad*bd*be*cf*ef+
                  ab*ad*cd*cf*ef+ab*ae*cd*cf*ef+ab*bd*cd*cf*ef+ae*bd*cd*cf*ef+ab*be*cd*cf*ef+ad*be*cd*cf*ef+
                  ab*ad*bc*de*ef+ac*ad*bc*de*ef+ab*af*bc*de*ef+ac*af*bc*de*ef+ab*ac*bd*de*ef+ac*af*bd*de*ef+
                  ac*bc*bd*de*ef+af*bc*bd*de*ef+ab*ac*bf*de*ef+ac*ad*bf*de*ef+ac*bc*bf*de*ef+ad*bc*bf*de*ef+
                  ab*ac*cd*de*ef+ab*af*cd*de*ef+ab*bc*cd*de*ef+af*bc*cd*de*ef+ab*bf*cd*de*ef+ac*bf*cd*de*ef+
                  ab*ac*cf*de*ef+ab*ad*cf*de*ef+ab*bc*cf*de*ef+ad*bc*cf*de*ef+ab*bd*cf*de*ef+ac*bd*cf*de*ef+
                  ab*ad*bc*df*ef+ac*ad*bc*df*ef+ab*ae*bc*df*ef+ac*ae*bc*df*ef+ab*ac*bd*df*ef+ac*ae*bd*df*ef+
                  ac*bc*bd*df*ef+ae*bc*bd*df*ef+ab*ac*be*df*ef+ac*ad*be*df*ef+ac*bc*be*df*ef+ad*bc*be*df*ef+
                  ab*ac*cd*df*ef+ab*ae*cd*df*ef+ab*bc*cd*df*ef+ae*bc*cd*df*ef+ab*be*cd*df*ef+ac*be*cd*df*ef+
                  ab*ac*ce*df*ef+ab*ad*ce*df*ef+ab*bc*ce*df*ef+ad*bc*ce*df*ef+ab*bd*ce*df*ef+ac*bd*ce*df*ef);
    }

    OverlapReal op6(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,
               OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,
               OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,
               OverlapReal de,OverlapReal df,OverlapReal dg,
               OverlapReal ef,OverlapReal eg,
               OverlapReal fg)
    {
        return 2*(ae*ag*bd*bf*cd*ce+af*ag*bd*bf*cd*ce+ad*ag*be*bf*cd*ce+af*ag*be*bf*cd*ce+ae*af*bd*bg*cd*ce+af*ag*bd*bg*cd*ce+
                  ad*af*be*bg*cd*ce+af*ag*be*bg*cd*ce+ad*af*bf*bg*cd*ce+ae*af*bf*bg*cd*ce+ad*ag*bf*bg*cd*ce+ae*ag*bf*bg*cd*ce+
                  ae*ag*bd*be*cd*cf+af*ag*bd*be*cd*cf+ad*ag*be*bf*cd*cf+ae*ag*be*bf*cd*cf+ae*af*bd*bg*cd*cf+ae*ag*bd*bg*cd*cf+
                  ad*ae*be*bg*cd*cf+ae*af*be*bg*cd*cf+ad*ag*be*bg*cd*cf+af*ag*be*bg*cd*cf+ad*ae*bf*bg*cd*cf+ae*ag*bf*bg*cd*cf+
                  ad*ag*bd*be*ce*cf+af*ag*bd*be*ce*cf+ad*ag*bd*bf*ce*cf+ae*ag*bd*bf*ce*cf+ad*ae*bd*bg*ce*cf+ad*af*bd*bg*ce*cf+
                  ae*ag*bd*bg*ce*cf+af*ag*bd*bg*ce*cf+ad*af*be*bg*ce*cf+ad*ag*be*bg*ce*cf+ad*ae*bf*bg*ce*cf+ad*ag*bf*bg*ce*cf+
                  ae*af*bd*be*cd*cg+af*ag*bd*be*cd*cg+ae*af*bd*bf*cd*cg+ae*ag*bd*bf*cd*cg+ad*ae*be*bf*cd*cg+ad*af*be*bf*cd*cg+
                  ae*ag*be*bf*cd*cg+af*ag*be*bf*cd*cg+ad*af*be*bg*cd*cg+ae*af*be*bg*cd*cg+ad*ae*bf*bg*cd*cg+ae*af*bf*bg*cd*cg+
                  ad*af*bd*be*ce*cg+af*ag*bd*be*ce*cg+ad*ae*bd*bf*ce*cg+ae*af*bd*bf*ce*cg+ad*ag*bd*bf*ce*cg+af*ag*bd*bf*ce*cg+
                  ad*af*be*bf*ce*cg+ad*ag*be*bf*ce*cg+ad*af*bd*bg*ce*cg+ae*af*bd*bg*ce*cg+ad*ae*bf*bg*ce*cg+ad*af*bf*bg*ce*cg+
                  ad*af*bd*be*cf*cg+ae*af*bd*be*cf*cg+ad*ag*bd*be*cf*cg+ae*ag*bd*be*cf*cg+ad*ae*bd*bf*cf*cg+ae*ag*bd*bf*cf*cg+
                  ad*ae*be*bf*cf*cg+ad*ag*be*bf*cf*cg+ad*ae*bd*bg*cf*cg+ae*af*bd*bg*cf*cg+ad*ae*be*bg*cf*cg+ad*af*be*bg*cf*cg+
                  ae*ag*bc*bf*cd*de+af*ag*bc*bf*cd*de+ac*ag*be*bf*cd*de+af*ag*be*bf*cd*de+ae*af*bc*bg*cd*de+af*ag*bc*bg*cd*de+
                  ac*af*be*bg*cd*de+af*ag*be*bg*cd*de+ac*af*bf*bg*cd*de+ae*af*bf*bg*cd*de+ac*ag*bf*bg*cd*de+ae*ag*bf*bg*cd*de+
                  ad*ag*bc*bf*ce*de+af*ag*bc*bf*ce*de+ac*ag*bd*bf*ce*de+af*ag*bd*bf*ce*de+ad*af*bc*bg*ce*de+af*ag*bc*bg*ce*de+
                  ac*af*bd*bg*ce*de+af*ag*bd*bg*ce*de+ac*af*bf*bg*ce*de+ad*af*bf*bg*ce*de+ac*ag*bf*bg*ce*de+ad*ag*bf*bg*ce*de+
                  ae*ag*bc*bd*cf*de+af*ag*bc*bd*cf*de+ad*ag*bc*be*cf*de+af*ag*bc*be*cf*de+ac*ag*bd*bf*cf*de+ae*ag*bd*bf*cf*de+
                  ac*ag*be*bf*cf*de+ad*ag*be*bf*cf*de+ad*af*bc*bg*cf*de+ae*af*bc*bg*cf*de+ad*ag*bc*bg*cf*de+ae*ag*bc*bg*cf*de+
                  ac*ae*bd*bg*cf*de+ae*af*bd*bg*cf*de+ac*ag*bd*bg*cf*de+af*ag*bd*bg*cf*de+ac*ad*be*bg*cf*de+ad*af*be*bg*cf*de+
                  ac*ag*be*bg*cf*de+af*ag*be*bg*cf*de+ac*ad*bf*bg*cf*de+ac*ae*bf*bg*cf*de+ad*ag*bf*bg*cf*de+ae*ag*bf*bg*cf*de+
                  ab*ag*be*cd*cf*de+af*ag*be*cd*cf*de+ab*ag*bf*cd*cf*de+ae*ag*bf*cd*cf*de+ab*ae*bg*cd*cf*de+ab*af*bg*cd*cf*de+
                  ae*ag*bg*cd*cf*de+af*ag*bg*cd*cf*de+af*be*bg*cd*cf*de+ag*be*bg*cd*cf*de+ae*bf*bg*cd*cf*de+ag*bf*bg*cd*cf*de+
                  ab*ag*bd*ce*cf*de+af*ag*bd*ce*cf*de+ab*ag*bf*ce*cf*de+ad*ag*bf*ce*cf*de+ab*ad*bg*ce*cf*de+ab*af*bg*ce*cf*de+
                  ad*ag*bg*ce*cf*de+af*ag*bg*ce*cf*de+af*bd*bg*ce*cf*de+ag*bd*bg*ce*cf*de+ad*bf*bg*ce*cf*de+ag*bf*bg*ce*cf*de+
                  ae*af*bc*bd*cg*de+af*ag*bc*bd*cg*de+ad*af*bc*be*cg*de+af*ag*bc*be*cg*de+ad*af*bc*bf*cg*de+ae*af*bc*bf*cg*de+
                  ad*ag*bc*bf*cg*de+ae*ag*bc*bf*cg*de+ac*ae*bd*bf*cg*de+ac*af*bd*bf*cg*de+ae*ag*bd*bf*cg*de+af*ag*bd*bf*cg*de+
                  ac*ad*be*bf*cg*de+ac*af*be*bf*cg*de+ad*ag*be*bf*cg*de+af*ag*be*bf*cg*de+ac*af*bd*bg*cg*de+ae*af*bd*bg*cg*de+
                  ac*af*be*bg*cg*de+ad*af*be*bg*cg*de+ac*ad*bf*bg*cg*de+ac*ae*bf*bg*cg*de+ad*af*bf*bg*cg*de+ae*af*bf*bg*cg*de+
                  ab*af*be*cd*cg*de+af*ag*be*cd*cg*de+ab*ae*bf*cd*cg*de+ae*af*bf*cd*cg*de+ab*ag*bf*cd*cg*de+af*ag*bf*cd*cg*de+
                  af*be*bf*cd*cg*de+ag*be*bf*cd*cg*de+ab*af*bg*cd*cg*de+ae*af*bg*cd*cg*de+ae*bf*bg*cd*cg*de+af*bf*bg*cd*cg*de+
                  ab*af*bd*ce*cg*de+af*ag*bd*ce*cg*de+ab*ad*bf*ce*cg*de+ad*af*bf*ce*cg*de+ab*ag*bf*ce*cg*de+af*ag*bf*ce*cg*de+
                  af*bd*bf*ce*cg*de+ag*bd*bf*ce*cg*de+ab*af*bg*ce*cg*de+ad*af*bg*ce*cg*de+ad*bf*bg*ce*cg*de+af*bf*bg*ce*cg*de+
                  ab*af*bd*cf*cg*de+ae*af*bd*cf*cg*de+ab*ag*bd*cf*cg*de+ae*ag*bd*cf*cg*de+ab*af*be*cf*cg*de+ad*af*be*cf*cg*de+
                  ab*ag*be*cf*cg*de+ad*ag*be*cf*cg*de+ab*ad*bf*cf*cg*de+ab*ae*bf*cf*cg*de+ad*ag*bf*cf*cg*de+ae*ag*bf*cf*cg*de+
                  ae*bd*bf*cf*cg*de+ag*bd*bf*cf*cg*de+ad*be*bf*cf*cg*de+ag*be*bf*cf*cg*de+ab*ad*bg*cf*cg*de+ab*ae*bg*cf*cg*de+
                  ad*af*bg*cf*cg*de+ae*af*bg*cf*cg*de+ae*bd*bg*cf*cg*de+af*bd*bg*cf*cg*de+ad*be*bg*cf*cg*de+af*be*bg*cf*cg*de+
                  ae*ag*bc*be*cd*df+af*ag*bc*be*cd*df+ac*ag*be*bf*cd*df+ae*ag*be*bf*cd*df+ae*af*bc*bg*cd*df+ae*ag*bc*bg*cd*df+
                  ac*ae*be*bg*cd*df+ae*af*be*bg*cd*df+ac*ag*be*bg*cd*df+af*ag*be*bg*cd*df+ac*ae*bf*bg*cd*df+ae*ag*bf*bg*cd*df+
                  ae*ag*bc*bd*ce*df+af*ag*bc*bd*ce*df+ac*ag*bd*be*ce*df+af*ag*bd*be*ce*df+ad*ag*bc*bf*ce*df+ae*ag*bc*bf*ce*df+
                  ac*ag*be*bf*ce*df+ad*ag*be*bf*ce*df+ad*ae*bc*bg*ce*df+ae*af*bc*bg*ce*df+ad*ag*bc*bg*ce*df+af*ag*bc*bg*ce*df+
                  ac*af*bd*bg*ce*df+ae*af*bd*bg*ce*df+ac*ag*bd*bg*ce*df+ae*ag*bd*bg*ce*df+ac*ad*be*bg*ce*df+ac*af*be*bg*ce*df+
                  ad*ag*be*bg*ce*df+af*ag*be*bg*ce*df+ac*ad*bf*bg*ce*df+ad*ae*bf*bg*ce*df+ac*ag*bf*bg*ce*df+ae*ag*bf*bg*ce*df+
                  ab*ag*be*cd*ce*df+af*ag*be*cd*ce*df+ab*ag*bf*cd*ce*df+ae*ag*bf*cd*ce*df+ab*ae*bg*cd*ce*df+ab*af*bg*cd*ce*df+
                  ae*ag*bg*cd*ce*df+af*ag*bg*cd*ce*df+af*be*bg*cd*ce*df+ag*be*bg*cd*ce*df+ae*bf*bg*cd*ce*df+ag*bf*bg*cd*ce*df+
                  ad*ag*bc*be*cf*df+ae*ag*bc*be*cf*df+ac*ag*bd*be*cf*df+ae*ag*bd*be*cf*df+ad*ae*bc*bg*cf*df+ae*ag*bc*bg*cf*df+
                  ac*ae*bd*bg*cf*df+ae*ag*bd*bg*cf*df+ac*ae*be*bg*cf*df+ad*ae*be*bg*cf*df+ac*ag*be*bg*cf*df+ad*ag*be*bg*cf*df+
                  ab*ag*bd*ce*cf*df+ae*ag*bd*ce*cf*df+ab*ag*be*ce*cf*df+ad*ag*be*ce*cf*df+ab*ad*bg*ce*cf*df+ab*ae*bg*ce*cf*df+
                  ad*ag*bg*ce*cf*df+ae*ag*bg*ce*cf*df+ae*bd*bg*ce*cf*df+ag*bd*bg*ce*cf*df+ad*be*bg*ce*cf*df+ag*be*bg*ce*cf*df+
                  ae*af*bc*bd*cg*df+ae*ag*bc*bd*cg*df+ad*ae*bc*be*cg*df+ae*af*bc*be*cg*df+ad*ag*bc*be*cg*df+af*ag*bc*be*cg*df+
                  ac*ae*bd*be*cg*df+ac*af*bd*be*cg*df+ae*ag*bd*be*cg*df+af*ag*bd*be*cg*df+ad*ae*bc*bf*cg*df+ae*ag*bc*bf*cg*df+
                  ac*ad*be*bf*cg*df+ac*ae*be*bf*cg*df+ad*ag*be*bf*cg*df+ae*ag*be*bf*cg*df+ac*ae*bd*bg*cg*df+ae*af*bd*bg*cg*df+
                  ac*ad*be*bg*cg*df+ad*ae*be*bg*cg*df+ac*af*be*bg*cg*df+ae*af*be*bg*cg*df+ac*ae*bf*bg*cg*df+ad*ae*bf*bg*cg*df+
                  ab*af*be*cd*cg*df+ae*af*be*cd*cg*df+ab*ag*be*cd*cg*df+ae*ag*be*cd*cg*df+ab*ae*bf*cd*cg*df+ae*ag*bf*cd*cg*df+
                  ae*be*bf*cd*cg*df+ag*be*bf*cd*cg*df+ab*ae*bg*cd*cg*df+ae*af*bg*cd*cg*df+ae*be*bg*cd*cg*df+af*be*bg*cd*cg*df+
                  ab*ae*bd*ce*cg*df+ae*af*bd*ce*cg*df+ab*ag*bd*ce*cg*df+af*ag*bd*ce*cg*df+ab*ad*be*ce*cg*df+ab*af*be*ce*cg*df+
                  ad*ag*be*ce*cg*df+af*ag*be*ce*cg*df+af*bd*be*ce*cg*df+ag*bd*be*ce*cg*df+ab*ae*bf*ce*cg*df+ad*ae*bf*ce*cg*df+
                  ab*ag*bf*ce*cg*df+ad*ag*bf*ce*cg*df+ad*be*bf*ce*cg*df+ag*be*bf*ce*cg*df+ab*ad*bg*ce*cg*df+ad*ae*bg*ce*cg*df+
                  ab*af*bg*ce*cg*df+ae*af*bg*ce*cg*df+ae*bd*bg*ce*cg*df+af*bd*bg*ce*cg*df+ad*bf*bg*ce*cg*df+ae*bf*bg*ce*cg*df+
                  ab*ae*bd*cf*cg*df+ae*ag*bd*cf*cg*df+ab*ad*be*cf*cg*df+ad*ae*be*cf*cg*df+ab*ag*be*cf*cg*df+ae*ag*be*cf*cg*df+
                  ae*bd*be*cf*cg*df+ag*bd*be*cf*cg*df+ab*ae*bg*cf*cg*df+ad*ae*bg*cf*cg*df+ad*be*bg*cf*cg*df+ae*be*bg*cf*cg*df+
                  ac*ag*bc*be*de*df+af*ag*bc*be*de*df+ac*ag*bc*bf*de*df+ae*ag*bc*bf*de*df+ac*ae*bc*bg*de*df+ac*af*bc*bg*de*df+
                  ae*ag*bc*bg*de*df+af*ag*bc*bg*de*df+ac*af*be*bg*de*df+ac*ag*be*bg*de*df+ac*ae*bf*bg*de*df+ac*ag*bf*bg*de*df+
                  ab*ag*bc*ce*de*df+af*ag*bc*ce*de*df+ab*ag*bf*ce*de*df+ac*ag*bf*ce*de*df+ab*ac*bg*ce*de*df+ab*af*bg*ce*de*df+
                  ac*ag*bg*ce*de*df+af*ag*bg*ce*de*df+af*bc*bg*ce*de*df+ag*bc*bg*ce*de*df+ac*bf*bg*ce*de*df+ag*bf*bg*ce*de*df+
                  ab*ag*bc*cf*de*df+ae*ag*bc*cf*de*df+ab*ag*be*cf*de*df+ac*ag*be*cf*de*df+ab*ac*bg*cf*de*df+ab*ae*bg*cf*de*df+
                  ac*ag*bg*cf*de*df+ae*ag*bg*cf*de*df+ae*bc*bg*cf*de*df+ag*bc*bg*cf*de*df+ac*be*bg*cf*de*df+ag*be*bg*cf*de*df+
                  ab*ae*bc*cg*de*df+ab*af*bc*cg*de*df+ae*ag*bc*cg*de*df+af*ag*bc*cg*de*df+ab*ac*be*cg*de*df+ac*af*be*cg*de*df+
                  ab*ag*be*cg*de*df+af*ag*be*cg*de*df+af*bc*be*cg*de*df+ag*bc*be*cg*de*df+ab*ac*bf*cg*de*df+ac*ae*bf*cg*de*df+
                  ab*ag*bf*cg*de*df+ae*ag*bf*cg*de*df+ae*bc*bf*cg*de*df+ag*bc*bf*cg*de*df+ab*ae*bg*cg*de*df+ac*ae*bg*cg*de*df+
                  ab*af*bg*cg*de*df+ac*af*bg*cg*de*df+ac*be*bg*cg*de*df+af*be*bg*cg*de*df+ac*bf*bg*cg*de*df+ae*bf*bg*cg*de*df+
                  ab*af*ce*cg*de*df+ab*ag*ce*cg*de*df+ab*bf*ce*cg*de*df+ag*bf*ce*cg*de*df+ab*bg*ce*cg*de*df+af*bg*ce*cg*de*df+
                  ab*ae*cf*cg*de*df+ab*ag*cf*cg*de*df+ab*be*cf*cg*de*df+ag*be*cf*cg*de*df+ab*bg*cf*cg*de*df+ae*bg*cf*cg*de*df+
                  ae*af*bc*be*cd*dg+af*ag*bc*be*cd*dg+ae*af*bc*bf*cd*dg+ae*ag*bc*bf*cd*dg+ac*ae*be*bf*cd*dg+ac*af*be*bf*cd*dg+
                  ae*ag*be*bf*cd*dg+af*ag*be*bf*cd*dg+ac*af*be*bg*cd*dg+ae*af*be*bg*cd*dg+ac*ae*bf*bg*cd*dg+ae*af*bf*bg*cd*dg+
                  ae*af*bc*bd*ce*dg+af*ag*bc*bd*ce*dg+ac*af*bd*be*ce*dg+af*ag*bd*be*ce*dg+ad*ae*bc*bf*ce*dg+ad*af*bc*bf*ce*dg+
                  ae*ag*bc*bf*ce*dg+af*ag*bc*bf*ce*dg+ac*af*bd*bf*ce*dg+ae*af*bd*bf*ce*dg+ac*ag*bd*bf*ce*dg+ae*ag*bd*bf*ce*dg+
                  ac*ad*be*bf*ce*dg+ad*af*be*bf*ce*dg+ac*ag*be*bf*ce*dg+af*ag*be*bf*ce*dg+ad*af*bc*bg*ce*dg+ae*af*bc*bg*ce*dg+
                  ac*af*be*bg*ce*dg+ad*af*be*bg*ce*dg+ac*ad*bf*bg*ce*dg+ad*ae*bf*bg*ce*dg+ac*af*bf*bg*ce*dg+ae*af*bf*bg*ce*dg+
                  ab*af*be*cd*ce*dg+af*ag*be*cd*ce*dg+ab*ae*bf*cd*ce*dg+ae*af*bf*cd*ce*dg+ab*ag*bf*cd*ce*dg+af*ag*bf*cd*ce*dg+
                  af*be*bf*cd*ce*dg+ag*be*bf*cd*ce*dg+ab*af*bg*cd*ce*dg+ae*af*bg*cd*ce*dg+ae*bf*bg*cd*ce*dg+af*bf*bg*cd*ce*dg+
                  ae*af*bc*bd*cf*dg+ae*ag*bc*bd*cf*dg+ad*ae*bc*be*cf*dg+ad*af*bc*be*cf*dg+ae*ag*bc*be*cf*dg+af*ag*bc*be*cf*dg+
                  ac*ae*bd*be*cf*dg+ae*af*bd*be*cf*dg+ac*ag*bd*be*cf*dg+af*ag*bd*be*cf*dg+ac*ae*bd*bf*cf*dg+ae*ag*bd*bf*cf*dg+
                  ac*ad*be*bf*cf*dg+ad*ae*be*bf*cf*dg+ac*ag*be*bf*cf*dg+ae*ag*be*bf*cf*dg+ad*ae*bc*bg*cf*dg+ae*af*bc*bg*cf*dg+
                  ac*ad*be*bg*cf*dg+ac*ae*be*bg*cf*dg+ad*af*be*bg*cf*dg+ae*af*be*bg*cf*dg+ac*ae*bf*bg*cf*dg+ad*ae*bf*bg*cf*dg+
                  ab*af*be*cd*cf*dg+ae*af*be*cd*cf*dg+ab*ag*be*cd*cf*dg+ae*ag*be*cd*cf*dg+ab*ae*bf*cd*cf*dg+ae*ag*bf*cd*cf*dg+
                  ae*be*bf*cd*cf*dg+ag*be*bf*cd*cf*dg+ab*ae*bg*cd*cf*dg+ae*af*bg*cd*cf*dg+ae*be*bg*cd*cf*dg+af*be*bg*cd*cf*dg+
                  ab*ae*bd*ce*cf*dg+ab*af*bd*ce*cf*dg+ae*ag*bd*ce*cf*dg+af*ag*bd*ce*cf*dg+ab*ad*be*ce*cf*dg+ad*af*be*ce*cf*dg+
                  ab*ag*be*ce*cf*dg+af*ag*be*ce*cf*dg+af*bd*be*ce*cf*dg+ag*bd*be*ce*cf*dg+ab*ad*bf*ce*cf*dg+ad*ae*bf*ce*cf*dg+
                  ab*ag*bf*ce*cf*dg+ae*ag*bf*ce*cf*dg+ae*bd*bf*ce*cf*dg+ag*bd*bf*ce*cf*dg+ab*ae*bg*ce*cf*dg+ad*ae*bg*ce*cf*dg+
                  ab*af*bg*ce*cf*dg+ad*af*bg*ce*cf*dg+ad*be*bg*ce*cf*dg+af*be*bg*ce*cf*dg+ad*bf*bg*ce*cf*dg+ae*bf*bg*ce*cf*dg+
                  ad*af*bc*be*cg*dg+ae*af*bc*be*cg*dg+ac*af*bd*be*cg*dg+ae*af*bd*be*cg*dg+ad*ae*bc*bf*cg*dg+ae*af*bc*bf*cg*dg+
                  ac*ae*bd*bf*cg*dg+ae*af*bd*bf*cg*dg+ac*ae*be*bf*cg*dg+ad*ae*be*bf*cg*dg+ac*af*be*bf*cg*dg+ad*af*be*bf*cg*dg+
                  ab*af*bd*ce*cg*dg+ae*af*bd*ce*cg*dg+ab*af*be*ce*cg*dg+ad*af*be*ce*cg*dg+ab*ad*bf*ce*cg*dg+ab*ae*bf*ce*cg*dg+
                  ad*af*bf*ce*cg*dg+ae*af*bf*ce*cg*dg+ae*bd*bf*ce*cg*dg+af*bd*bf*ce*cg*dg+ad*be*bf*ce*cg*dg+af*be*bf*ce*cg*dg+
                  ab*ae*bd*cf*cg*dg+ae*af*bd*cf*cg*dg+ab*ad*be*cf*cg*dg+ad*ae*be*cf*cg*dg+ab*af*be*cf*cg*dg+ae*af*be*cf*cg*dg+
                  ae*bd*be*cf*cg*dg+af*bd*be*cf*cg*dg+ab*ae*bf*cf*cg*dg+ad*ae*bf*cf*cg*dg+ad*be*bf*cf*cg*dg+ae*be*bf*cf*cg*dg+
                  ac*af*bc*be*de*dg+af*ag*bc*be*de*dg+ac*ae*bc*bf*de*dg+ae*af*bc*bf*de*dg+ac*ag*bc*bf*de*dg+af*ag*bc*bf*de*dg+
                  ac*af*be*bf*de*dg+ac*ag*be*bf*de*dg+ac*af*bc*bg*de*dg+ae*af*bc*bg*de*dg+ac*ae*bf*bg*de*dg+ac*af*bf*bg*de*dg+
                  ab*af*bc*ce*de*dg+af*ag*bc*ce*de*dg+ab*ac*bf*ce*de*dg+ac*af*bf*ce*de*dg+ab*ag*bf*ce*de*dg+af*ag*bf*ce*de*dg+
                  af*bc*bf*ce*de*dg+ag*bc*bf*ce*de*dg+ab*af*bg*ce*de*dg+ac*af*bg*ce*de*dg+ac*bf*bg*ce*de*dg+af*bf*bg*ce*de*dg+
                  ab*ae*bc*cf*de*dg+ae*af*bc*cf*de*dg+ab*ag*bc*cf*de*dg+af*ag*bc*cf*de*dg+ab*ac*be*cf*de*dg+ab*af*be*cf*de*dg+
                  ac*ag*be*cf*de*dg+af*ag*be*cf*de*dg+af*bc*be*cf*de*dg+ag*bc*be*cf*de*dg+ab*ae*bf*cf*de*dg+ac*ae*bf*cf*de*dg+
                  ab*ag*bf*cf*de*dg+ac*ag*bf*cf*de*dg+ac*be*bf*cf*de*dg+ag*be*bf*cf*de*dg+ab*ac*bg*cf*de*dg+ac*ae*bg*cf*de*dg+
                  ab*af*bg*cf*de*dg+ae*af*bg*cf*de*dg+ae*bc*bg*cf*de*dg+af*bc*bg*cf*de*dg+ac*bf*bg*cf*de*dg+ae*bf*bg*cf*de*dg+
                  ab*af*ce*cf*de*dg+ab*ag*ce*cf*de*dg+ab*bf*ce*cf*de*dg+ag*bf*ce*cf*de*dg+ab*bg*ce*cf*de*dg+af*bg*ce*cf*de*dg+
                  ab*af*bc*cg*de*dg+ae*af*bc*cg*de*dg+ab*af*be*cg*de*dg+ac*af*be*cg*de*dg+ab*ac*bf*cg*de*dg+ab*ae*bf*cg*de*dg+
                  ac*af*bf*cg*de*dg+ae*af*bf*cg*de*dg+ae*bc*bf*cg*de*dg+af*bc*bf*cg*de*dg+ac*be*bf*cg*de*dg+af*be*bf*cg*de*dg+
                  ab*ae*cf*cg*de*dg+ab*af*cf*cg*de*dg+ab*be*cf*cg*de*dg+af*be*cf*cg*de*dg+ab*bf*cf*cg*de*dg+ae*bf*cf*cg*de*dg+
                  ac*af*bc*be*df*dg+ae*af*bc*be*df*dg+ac*ag*bc*be*df*dg+ae*ag*bc*be*df*dg+ac*ae*bc*bf*df*dg+ae*ag*bc*bf*df*dg+
                  ac*ae*be*bf*df*dg+ac*ag*be*bf*df*dg+ac*ae*bc*bg*df*dg+ae*af*bc*bg*df*dg+ac*ae*be*bg*df*dg+ac*af*be*bg*df*dg+
                  ab*af*bc*ce*df*dg+ae*af*bc*ce*df*dg+ab*ag*bc*ce*df*dg+ae*ag*bc*ce*df*dg+ab*af*be*ce*df*dg+ac*af*be*ce*df*dg+
                  ab*ag*be*ce*df*dg+ac*ag*be*ce*df*dg+ab*ac*bf*ce*df*dg+ab*ae*bf*ce*df*dg+ac*ag*bf*ce*df*dg+ae*ag*bf*ce*df*dg+
                  ae*bc*bf*ce*df*dg+ag*bc*bf*ce*df*dg+ac*be*bf*ce*df*dg+ag*be*bf*ce*df*dg+ab*ac*bg*ce*df*dg+ab*ae*bg*ce*df*dg+
                  ac*af*bg*ce*df*dg+ae*af*bg*ce*df*dg+ae*bc*bg*ce*df*dg+af*bc*bg*ce*df*dg+ac*be*bg*ce*df*dg+af*be*bg*ce*df*dg+
                  ab*ae*bc*cf*df*dg+ae*ag*bc*cf*df*dg+ab*ac*be*cf*df*dg+ac*ae*be*cf*df*dg+ab*ag*be*cf*df*dg+ae*ag*be*cf*df*dg+
                  ae*bc*be*cf*df*dg+ag*bc*be*cf*df*dg+ab*ae*bg*cf*df*dg+ac*ae*bg*cf*df*dg+ac*be*bg*cf*df*dg+ae*be*bg*cf*df*dg+
                  ab*ae*ce*cf*df*dg+ab*ag*ce*cf*df*dg+ab*be*ce*cf*df*dg+ag*be*ce*cf*df*dg+ab*bg*ce*cf*df*dg+ae*bg*ce*cf*df*dg+
                  ab*ae*bc*cg*df*dg+ae*af*bc*cg*df*dg+ab*ac*be*cg*df*dg+ac*ae*be*cg*df*dg+ab*af*be*cg*df*dg+ae*af*be*cg*df*dg+
                  ae*bc*be*cg*df*dg+af*bc*be*cg*df*dg+ab*ae*bf*cg*df*dg+ac*ae*bf*cg*df*dg+ac*be*bf*cg*df*dg+ae*be*bf*cg*df*dg+
                  ab*ae*ce*cg*df*dg+ab*af*ce*cg*df*dg+ab*be*ce*cg*df*dg+af*be*ce*cg*df*dg+ab*bf*ce*cg*df*dg+ae*bf*ce*cg*df*dg+
                  ad*ag*bc*be*cd*ef+af*ag*bc*be*cd*ef+ac*ag*bd*be*cd*ef+af*ag*bd*be*cd*ef+ad*ag*bc*bf*cd*ef+ae*ag*bc*bf*cd*ef+
                  ac*ag*bd*bf*cd*ef+ae*ag*bd*bf*cd*ef+ad*ae*bc*bg*cd*ef+ad*af*bc*bg*cd*ef+ae*ag*bc*bg*cd*ef+af*ag*bc*bg*cd*ef+
                  ac*ae*bd*bg*cd*ef+ac*af*bd*bg*cd*ef+ae*ag*bd*bg*cd*ef+af*ag*bd*bg*cd*ef+ac*af*be*bg*cd*ef+ad*af*be*bg*cd*ef+
                  ac*ag*be*bg*cd*ef+ad*ag*be*bg*cd*ef+ac*ae*bf*bg*cd*ef+ad*ae*bf*bg*cd*ef+ac*ag*bf*bg*cd*ef+ad*ag*bf*bg*cd*ef+
                  ad*ag*bc*bd*ce*ef+af*ag*bc*bd*ce*ef+ac*ag*bd*bf*ce*ef+ad*ag*bd*bf*ce*ef+ad*af*bc*bg*ce*ef+ad*ag*bc*bg*ce*ef+
                  ac*ad*bd*bg*ce*ef+ad*af*bd*bg*ce*ef+ac*ag*bd*bg*ce*ef+af*ag*bd*bg*ce*ef+ac*ad*bf*bg*ce*ef+ad*ag*bf*bg*ce*ef+
                  ab*ag*bd*cd*ce*ef+af*ag*bd*cd*ce*ef+ab*ag*bf*cd*ce*ef+ad*ag*bf*cd*ce*ef+ab*ad*bg*cd*ce*ef+ab*af*bg*cd*ce*ef+
                  ad*ag*bg*cd*ce*ef+af*ag*bg*cd*ce*ef+af*bd*bg*cd*ce*ef+ag*bd*bg*cd*ce*ef+ad*bf*bg*cd*ce*ef+ag*bf*bg*cd*ce*ef+
                  ad*ag*bc*bd*cf*ef+ae*ag*bc*bd*cf*ef+ac*ag*bd*be*cf*ef+ad*ag*bd*be*cf*ef+ad*ae*bc*bg*cf*ef+ad*ag*bc*bg*cf*ef+
                  ac*ad*bd*bg*cf*ef+ad*ae*bd*bg*cf*ef+ac*ag*bd*bg*cf*ef+ae*ag*bd*bg*cf*ef+ac*ad*be*bg*cf*ef+ad*ag*be*bg*cf*ef+
                  ab*ag*bd*cd*cf*ef+ae*ag*bd*cd*cf*ef+ab*ag*be*cd*cf*ef+ad*ag*be*cd*cf*ef+ab*ad*bg*cd*cf*ef+ab*ae*bg*cd*cf*ef+
                  ad*ag*bg*cd*cf*ef+ae*ag*bg*cd*cf*ef+ae*bd*bg*cd*cf*ef+ag*bd*bg*cd*cf*ef+ad*be*bg*cd*cf*ef+ag*be*bg*cd*cf*ef+
                  ad*ae*bc*bd*cg*ef+ad*af*bc*bd*cg*ef+ae*ag*bc*bd*cg*ef+af*ag*bc*bd*cg*ef+ad*af*bc*be*cg*ef+ad*ag*bc*be*cg*ef+
                  ac*ad*bd*be*cg*ef+ac*af*bd*be*cg*ef+ad*ag*bd*be*cg*ef+af*ag*bd*be*cg*ef+ad*ae*bc*bf*cg*ef+ad*ag*bc*bf*cg*ef+
                  ac*ad*bd*bf*cg*ef+ac*ae*bd*bf*cg*ef+ad*ag*bd*bf*cg*ef+ae*ag*bd*bf*cg*ef+ac*ae*bd*bg*cg*ef+ad*ae*bd*bg*cg*ef+
                  ac*af*bd*bg*cg*ef+ad*af*bd*bg*cg*ef+ac*ad*be*bg*cg*ef+ad*af*be*bg*cg*ef+ac*ad*bf*bg*cg*ef+ad*ae*bf*bg*cg*ef+
                  ab*ae*bd*cd*cg*ef+ab*af*bd*cd*cg*ef+ae*ag*bd*cd*cg*ef+af*ag*bd*cd*cg*ef+ab*ad*be*cd*cg*ef+ad*af*be*cd*cg*ef+
                  ab*ag*be*cd*cg*ef+af*ag*be*cd*cg*ef+af*bd*be*cd*cg*ef+ag*bd*be*cd*cg*ef+ab*ad*bf*cd*cg*ef+ad*ae*bf*cd*cg*ef+
                  ab*ag*bf*cd*cg*ef+ae*ag*bf*cd*cg*ef+ae*bd*bf*cd*cg*ef+ag*bd*bf*cd*cg*ef+ab*ae*bg*cd*cg*ef+ad*ae*bg*cd*cg*ef+
                  ab*af*bg*cd*cg*ef+ad*af*bg*cd*cg*ef+ad*be*bg*cd*cg*ef+af*be*bg*cd*cg*ef+ad*bf*bg*cd*cg*ef+ae*bf*bg*cd*cg*ef+
                  ab*af*bd*ce*cg*ef+ad*af*bd*ce*cg*ef+ab*ag*bd*ce*cg*ef+ad*ag*bd*ce*cg*ef+ab*ad*bf*ce*cg*ef+ad*ag*bf*ce*cg*ef+
                  ad*bd*bf*ce*cg*ef+ag*bd*bf*ce*cg*ef+ab*ad*bg*ce*cg*ef+ad*af*bg*ce*cg*ef+ad*bd*bg*ce*cg*ef+af*bd*bg*ce*cg*ef+
                  ab*ae*bd*cf*cg*ef+ad*ae*bd*cf*cg*ef+ab*ag*bd*cf*cg*ef+ad*ag*bd*cf*cg*ef+ab*ad*be*cf*cg*ef+ad*ag*be*cf*cg*ef+
                  ad*bd*be*cf*cg*ef+ag*bd*be*cf*cg*ef+ab*ad*bg*cf*cg*ef+ad*ae*bg*cf*cg*ef+ad*bd*bg*cf*cg*ef+ae*bd*bg*cf*cg*ef+
                  ac*ag*bc*bd*de*ef+af*ag*bc*bd*de*ef+ac*ag*bc*bf*de*ef+ad*ag*bc*bf*de*ef+ac*ad*bc*bg*de*ef+ac*af*bc*bg*de*ef+
                  ad*ag*bc*bg*de*ef+af*ag*bc*bg*de*ef+ac*af*bd*bg*de*ef+ac*ag*bd*bg*de*ef+ac*ad*bf*bg*de*ef+ac*ag*bf*bg*de*ef+
                  ab*ag*bc*cd*de*ef+af*ag*bc*cd*de*ef+ab*ag*bf*cd*de*ef+ac*ag*bf*cd*de*ef+ab*ac*bg*cd*de*ef+ab*af*bg*cd*de*ef+
                  ac*ag*bg*cd*de*ef+af*ag*bg*cd*de*ef+af*bc*bg*cd*de*ef+ag*bc*bg*cd*de*ef+ac*bf*bg*cd*de*ef+ag*bf*bg*cd*de*ef+
                  ab*ag*bc*cf*de*ef+ad*ag*bc*cf*de*ef+ab*ag*bd*cf*de*ef+ac*ag*bd*cf*de*ef+ab*ac*bg*cf*de*ef+ab*ad*bg*cf*de*ef+
                  ac*ag*bg*cf*de*ef+ad*ag*bg*cf*de*ef+ad*bc*bg*cf*de*ef+ag*bc*bg*cf*de*ef+ac*bd*bg*cf*de*ef+ag*bd*bg*cf*de*ef+
                  ab*ad*bc*cg*de*ef+ab*af*bc*cg*de*ef+ad*ag*bc*cg*de*ef+af*ag*bc*cg*de*ef+ab*ac*bd*cg*de*ef+ac*af*bd*cg*de*ef+
                  ab*ag*bd*cg*de*ef+af*ag*bd*cg*de*ef+af*bc*bd*cg*de*ef+ag*bc*bd*cg*de*ef+ab*ac*bf*cg*de*ef+ac*ad*bf*cg*de*ef+
                  ab*ag*bf*cg*de*ef+ad*ag*bf*cg*de*ef+ad*bc*bf*cg*de*ef+ag*bc*bf*cg*de*ef+ab*ad*bg*cg*de*ef+ac*ad*bg*cg*de*ef+
                  ab*af*bg*cg*de*ef+ac*af*bg*cg*de*ef+ac*bd*bg*cg*de*ef+af*bd*bg*cg*de*ef+ac*bf*bg*cg*de*ef+ad*bf*bg*cg*de*ef+
                  ab*af*cd*cg*de*ef+ab*ag*cd*cg*de*ef+ab*bf*cd*cg*de*ef+ag*bf*cd*cg*de*ef+ab*bg*cd*cg*de*ef+af*bg*cd*cg*de*ef+
                  ab*ad*cf*cg*de*ef+ab*ag*cf*cg*de*ef+ab*bd*cf*cg*de*ef+ag*bd*cf*cg*de*ef+ab*bg*cf*cg*de*ef+ad*bg*cf*cg*de*ef+
                  ac*ag*bc*bd*df*ef+ae*ag*bc*bd*df*ef+ac*ag*bc*be*df*ef+ad*ag*bc*be*df*ef+ac*ad*bc*bg*df*ef+ac*ae*bc*bg*df*ef+
                  ad*ag*bc*bg*df*ef+ae*ag*bc*bg*df*ef+ac*ae*bd*bg*df*ef+ac*ag*bd*bg*df*ef+ac*ad*be*bg*df*ef+ac*ag*be*bg*df*ef+
                  ab*ag*bc*cd*df*ef+ae*ag*bc*cd*df*ef+ab*ag*be*cd*df*ef+ac*ag*be*cd*df*ef+ab*ac*bg*cd*df*ef+ab*ae*bg*cd*df*ef+
                  ac*ag*bg*cd*df*ef+ae*ag*bg*cd*df*ef+ae*bc*bg*cd*df*ef+ag*bc*bg*cd*df*ef+ac*be*bg*cd*df*ef+ag*be*bg*cd*df*ef+
                  ab*ag*bc*ce*df*ef+ad*ag*bc*ce*df*ef+ab*ag*bd*ce*df*ef+ac*ag*bd*ce*df*ef+ab*ac*bg*ce*df*ef+ab*ad*bg*ce*df*ef+
                  ac*ag*bg*ce*df*ef+ad*ag*bg*ce*df*ef+ad*bc*bg*ce*df*ef+ag*bc*bg*ce*df*ef+ac*bd*bg*ce*df*ef+ag*bd*bg*ce*df*ef+
                  ab*ad*bc*cg*df*ef+ab*ae*bc*cg*df*ef+ad*ag*bc*cg*df*ef+ae*ag*bc*cg*df*ef+ab*ac*bd*cg*df*ef+ac*ae*bd*cg*df*ef+
                  ab*ag*bd*cg*df*ef+ae*ag*bd*cg*df*ef+ae*bc*bd*cg*df*ef+ag*bc*bd*cg*df*ef+ab*ac*be*cg*df*ef+ac*ad*be*cg*df*ef+
                  ab*ag*be*cg*df*ef+ad*ag*be*cg*df*ef+ad*bc*be*cg*df*ef+ag*bc*be*cg*df*ef+ab*ad*bg*cg*df*ef+ac*ad*bg*cg*df*ef+
                  ab*ae*bg*cg*df*ef+ac*ae*bg*cg*df*ef+ac*bd*bg*cg*df*ef+ae*bd*bg*cg*df*ef+ac*be*bg*cg*df*ef+ad*be*bg*cg*df*ef+
                  ab*ae*cd*cg*df*ef+ab*ag*cd*cg*df*ef+ab*be*cd*cg*df*ef+ag*be*cd*cg*df*ef+ab*bg*cd*cg*df*ef+ae*bg*cd*cg*df*ef+
                  ab*ad*ce*cg*df*ef+ab*ag*ce*cg*df*ef+ab*bd*ce*cg*df*ef+ag*bd*ce*cg*df*ef+ab*bg*ce*cg*df*ef+ad*bg*ce*cg*df*ef+
                  ac*ae*bc*bd*dg*ef+ac*af*bc*bd*dg*ef+ae*ag*bc*bd*dg*ef+af*ag*bc*bd*dg*ef+ac*ad*bc*be*dg*ef+ad*af*bc*be*dg*ef+
                  ac*ag*bc*be*dg*ef+af*ag*bc*be*dg*ef+ac*af*bd*be*dg*ef+ac*ag*bd*be*dg*ef+ac*ad*bc*bf*dg*ef+ad*ae*bc*bf*dg*ef+
                  ac*ag*bc*bf*dg*ef+ae*ag*bc*bf*dg*ef+ac*ae*bd*bf*dg*ef+ac*ag*bd*bf*dg*ef+ac*ae*bc*bg*dg*ef+ad*ae*bc*bg*dg*ef+
                  ac*af*bc*bg*dg*ef+ad*af*bc*bg*dg*ef+ac*ad*be*bg*dg*ef+ac*af*be*bg*dg*ef+ac*ad*bf*bg*dg*ef+ac*ae*bf*bg*dg*ef+
                  ab*ae*bc*cd*dg*ef+ab*af*bc*cd*dg*ef+ae*ag*bc*cd*dg*ef+af*ag*bc*cd*dg*ef+ab*ac*be*cd*dg*ef+ac*af*be*cd*dg*ef+
                  ab*ag*be*cd*dg*ef+af*ag*be*cd*dg*ef+af*bc*be*cd*dg*ef+ag*bc*be*cd*dg*ef+ab*ac*bf*cd*dg*ef+ac*ae*bf*cd*dg*ef+
                  ab*ag*bf*cd*dg*ef+ae*ag*bf*cd*dg*ef+ae*bc*bf*cd*dg*ef+ag*bc*bf*cd*dg*ef+ab*ae*bg*cd*dg*ef+ac*ae*bg*cd*dg*ef+
                  ab*af*bg*cd*dg*ef+ac*af*bg*cd*dg*ef+ac*be*bg*cd*dg*ef+af*be*bg*cd*dg*ef+ac*bf*bg*cd*dg*ef+ae*bf*bg*cd*dg*ef+
                  ab*ad*bc*ce*dg*ef+ad*af*bc*ce*dg*ef+ab*ag*bc*ce*dg*ef+af*ag*bc*ce*dg*ef+ab*ac*bd*ce*dg*ef+ab*af*bd*ce*dg*ef+
                  ac*ag*bd*ce*dg*ef+af*ag*bd*ce*dg*ef+af*bc*bd*ce*dg*ef+ag*bc*bd*ce*dg*ef+ab*ad*bf*ce*dg*ef+ac*ad*bf*ce*dg*ef+
                  ab*ag*bf*ce*dg*ef+ac*ag*bf*ce*dg*ef+ac*bd*bf*ce*dg*ef+ag*bd*bf*ce*dg*ef+ab*ac*bg*ce*dg*ef+ac*ad*bg*ce*dg*ef+
                  ab*af*bg*ce*dg*ef+ad*af*bg*ce*dg*ef+ad*bc*bg*ce*dg*ef+af*bc*bg*ce*dg*ef+ac*bf*bg*ce*dg*ef+ad*bf*bg*ce*dg*ef+
                  ab*af*cd*ce*dg*ef+ab*ag*cd*ce*dg*ef+ab*bf*cd*ce*dg*ef+ag*bf*cd*ce*dg*ef+ab*bg*cd*ce*dg*ef+af*bg*cd*ce*dg*ef+
                  ab*ad*bc*cf*dg*ef+ad*ae*bc*cf*dg*ef+ab*ag*bc*cf*dg*ef+ae*ag*bc*cf*dg*ef+ab*ac*bd*cf*dg*ef+ab*ae*bd*cf*dg*ef+
                  ac*ag*bd*cf*dg*ef+ae*ag*bd*cf*dg*ef+ae*bc*bd*cf*dg*ef+ag*bc*bd*cf*dg*ef+ab*ad*be*cf*dg*ef+ac*ad*be*cf*dg*ef+
                  ab*ag*be*cf*dg*ef+ac*ag*be*cf*dg*ef+ac*bd*be*cf*dg*ef+ag*bd*be*cf*dg*ef+ab*ac*bg*cf*dg*ef+ac*ad*bg*cf*dg*ef+
                  ab*ae*bg*cf*dg*ef+ad*ae*bg*cf*dg*ef+ad*bc*bg*cf*dg*ef+ae*bc*bg*cf*dg*ef+ac*be*bg*cf*dg*ef+ad*be*bg*cf*dg*ef+
                  ab*ae*cd*cf*dg*ef+ab*ag*cd*cf*dg*ef+ab*be*cd*cf*dg*ef+ag*be*cd*cf*dg*ef+ab*bg*cd*cf*dg*ef+ae*bg*cd*cf*dg*ef+
                  ab*ae*bc*cg*dg*ef+ad*ae*bc*cg*dg*ef+ab*af*bc*cg*dg*ef+ad*af*bc*cg*dg*ef+ab*ae*bd*cg*dg*ef+ac*ae*bd*cg*dg*ef+
                  ab*af*bd*cg*dg*ef+ac*af*bd*cg*dg*ef+ab*ac*be*cg*dg*ef+ab*ad*be*cg*dg*ef+ac*af*be*cg*dg*ef+ad*af*be*cg*dg*ef+
                  ad*bc*be*cg*dg*ef+af*bc*be*cg*dg*ef+ac*bd*be*cg*dg*ef+af*bd*be*cg*dg*ef+ab*ac*bf*cg*dg*ef+ab*ad*bf*cg*dg*ef+
                  ac*ae*bf*cg*dg*ef+ad*ae*bf*cg*dg*ef+ad*bc*bf*cg*dg*ef+ae*bc*bf*cg*dg*ef+ac*bd*bf*cg*dg*ef+ae*bd*bf*cg*dg*ef+
                  ab*ad*ce*cg*dg*ef+ab*af*ce*cg*dg*ef+ab*bd*ce*cg*dg*ef+af*bd*ce*cg*dg*ef+ab*bf*ce*cg*dg*ef+ad*bf*ce*cg*dg*ef+
                  ab*ad*cf*cg*dg*ef+ab*ae*cf*cg*dg*ef+ab*bd*cf*cg*dg*ef+ae*bd*cf*cg*dg*ef+ab*be*cf*cg*dg*ef+ad*be*cf*cg*dg*ef+
                  ab*af*bc*de*dg*ef+ac*af*bc*de*dg*ef+ab*ag*bc*de*dg*ef+ac*ag*bc*de*dg*ef+ab*ac*bf*de*dg*ef+ac*ag*bf*de*dg*ef+
                  ac*bc*bf*de*dg*ef+ag*bc*bf*de*dg*ef+ab*ac*bg*de*dg*ef+ac*af*bg*de*dg*ef+ac*bc*bg*de*dg*ef+af*bc*bg*de*dg*ef+
                  ab*ac*cf*de*dg*ef+ab*ag*cf*de*dg*ef+ab*bc*cf*de*dg*ef+ag*bc*cf*de*dg*ef+ab*bg*cf*de*dg*ef+ac*bg*cf*de*dg*ef+
                  ab*ac*cg*de*dg*ef+ab*af*cg*de*dg*ef+ab*bc*cg*de*dg*ef+af*bc*cg*de*dg*ef+ab*bf*cg*de*dg*ef+ac*bf*cg*de*dg*ef+
                  ab*ae*bc*df*dg*ef+ac*ae*bc*df*dg*ef+ab*ag*bc*df*dg*ef+ac*ag*bc*df*dg*ef+ab*ac*be*df*dg*ef+ac*ag*be*df*dg*ef+
                  ac*bc*be*df*dg*ef+ag*bc*be*df*dg*ef+ab*ac*bg*df*dg*ef+ac*ae*bg*df*dg*ef+ac*bc*bg*df*dg*ef+ae*bc*bg*df*dg*ef+
                  ab*ac*ce*df*dg*ef+ab*ag*ce*df*dg*ef+ab*bc*ce*df*dg*ef+ag*bc*ce*df*dg*ef+ab*bg*ce*df*dg*ef+ac*bg*ce*df*dg*ef+
                  ab*ac*cg*df*dg*ef+ab*ae*cg*df*dg*ef+ab*bc*cg*df*dg*ef+ae*bc*cg*df*dg*ef+ab*be*cg*df*dg*ef+ac*be*cg*df*dg*ef+
                  ad*af*bc*be*cd*eg+af*ag*bc*be*cd*eg+ac*af*bd*be*cd*eg+af*ag*bd*be*cd*eg+ad*ae*bc*bf*cd*eg+ae*af*bc*bf*cd*eg+
                  ad*ag*bc*bf*cd*eg+af*ag*bc*bf*cd*eg+ac*ae*bd*bf*cd*eg+ae*af*bd*bf*cd*eg+ac*ag*bd*bf*cd*eg+af*ag*bd*bf*cd*eg+
                  ac*af*be*bf*cd*eg+ad*af*be*bf*cd*eg+ac*ag*be*bf*cd*eg+ad*ag*be*bf*cd*eg+ad*af*bc*bg*cd*eg+ae*af*bc*bg*cd*eg+
                  ac*af*bd*bg*cd*eg+ae*af*bd*bg*cd*eg+ac*ae*bf*bg*cd*eg+ad*ae*bf*bg*cd*eg+ac*af*bf*bg*cd*eg+ad*af*bf*bg*cd*eg+
                  ad*af*bc*bd*ce*eg+af*ag*bc*bd*ce*eg+ad*af*bc*bf*ce*eg+ad*ag*bc*bf*ce*eg+ac*ad*bd*bf*ce*eg+ac*af*bd*bf*ce*eg+
                  ad*ag*bd*bf*ce*eg+af*ag*bd*bf*ce*eg+ac*af*bd*bg*ce*eg+ad*af*bd*bg*ce*eg+ac*ad*bf*bg*ce*eg+ad*af*bf*bg*ce*eg+
                  ab*af*bd*cd*ce*eg+af*ag*bd*cd*ce*eg+ab*ad*bf*cd*ce*eg+ad*af*bf*cd*ce*eg+ab*ag*bf*cd*ce*eg+af*ag*bf*cd*ce*eg+
                  af*bd*bf*cd*ce*eg+ag*bd*bf*cd*ce*eg+ab*af*bg*cd*ce*eg+ad*af*bg*cd*ce*eg+ad*bf*bg*cd*ce*eg+af*bf*bg*cd*ce*eg+
                  ad*ae*bc*bd*cf*eg+ae*af*bc*bd*cf*eg+ad*ag*bc*bd*cf*eg+af*ag*bc*bd*cf*eg+ad*af*bc*be*cf*eg+ad*ag*bc*be*cf*eg+
                  ac*ad*bd*be*cf*eg+ad*af*bd*be*cf*eg+ac*ag*bd*be*cf*eg+af*ag*bd*be*cf*eg+ac*ae*bd*bf*cf*eg+ad*ae*bd*bf*cf*eg+
                  ac*ag*bd*bf*cf*eg+ad*ag*bd*bf*cf*eg+ac*ad*be*bf*cf*eg+ad*ag*be*bf*cf*eg+ad*ae*bc*bg*cf*eg+ad*af*bc*bg*cf*eg+
                  ac*ad*bd*bg*cf*eg+ac*ae*bd*bg*cf*eg+ad*af*bd*bg*cf*eg+ae*af*bd*bg*cf*eg+ac*ad*bf*bg*cf*eg+ad*ae*bf*bg*cf*eg+
                  ab*ae*bd*cd*cf*eg+ae*af*bd*cd*cf*eg+ab*ag*bd*cd*cf*eg+af*ag*bd*cd*cf*eg+ab*ad*be*cd*cf*eg+ab*af*be*cd*cf*eg+
                  ad*ag*be*cd*cf*eg+af*ag*be*cd*cf*eg+af*bd*be*cd*cf*eg+ag*bd*be*cd*cf*eg+ab*ae*bf*cd*cf*eg+ad*ae*bf*cd*cf*eg+
                  ab*ag*bf*cd*cf*eg+ad*ag*bf*cd*cf*eg+ad*be*bf*cd*cf*eg+ag*be*bf*cd*cf*eg+ab*ad*bg*cd*cf*eg+ad*ae*bg*cd*cf*eg+
                  ab*af*bg*cd*cf*eg+ae*af*bg*cd*cf*eg+ae*bd*bg*cd*cf*eg+af*bd*bg*cd*cf*eg+ad*bf*bg*cd*cf*eg+ae*bf*bg*cd*cf*eg+
                  ab*af*bd*ce*cf*eg+ad*af*bd*ce*cf*eg+ab*ag*bd*ce*cf*eg+ad*ag*bd*ce*cf*eg+ab*ad*bf*ce*cf*eg+ad*ag*bf*ce*cf*eg+
                  ad*bd*bf*ce*cf*eg+ag*bd*bf*ce*cf*eg+ab*ad*bg*ce*cf*eg+ad*af*bg*ce*cf*eg+ad*bd*bg*ce*cf*eg+af*bd*bg*ce*cf*eg+
                  ad*af*bc*bd*cg*eg+ae*af*bc*bd*cg*eg+ac*af*bd*be*cg*eg+ad*af*bd*be*cg*eg+ad*ae*bc*bf*cg*eg+ad*af*bc*bf*cg*eg+
                  ac*ad*bd*bf*cg*eg+ad*ae*bd*bf*cg*eg+ac*af*bd*bf*cg*eg+ae*af*bd*bf*cg*eg+ac*ad*be*bf*cg*eg+ad*af*be*bf*cg*eg+
                  ab*af*bd*cd*cg*eg+ae*af*bd*cd*cg*eg+ab*af*be*cd*cg*eg+ad*af*be*cd*cg*eg+ab*ad*bf*cd*cg*eg+ab*ae*bf*cd*cg*eg+
                  ad*af*bf*cd*cg*eg+ae*af*bf*cd*cg*eg+ae*bd*bf*cd*cg*eg+af*bd*bf*cd*cg*eg+ad*be*bf*cd*cg*eg+af*be*bf*cd*cg*eg+
                  ab*ae*bd*cf*cg*eg+ad*ae*bd*cf*cg*eg+ab*af*bd*cf*cg*eg+ad*af*bd*cf*cg*eg+ab*ad*be*cf*cg*eg+ad*af*be*cf*cg*eg+
                  ad*bd*be*cf*cg*eg+af*bd*be*cf*cg*eg+ab*ad*bf*cf*cg*eg+ad*ae*bf*cf*cg*eg+ad*bd*bf*cf*cg*eg+ae*bd*bf*cf*cg*eg+
                  ac*af*bc*bd*de*eg+af*ag*bc*bd*de*eg+ac*ad*bc*bf*de*eg+ad*af*bc*bf*de*eg+ac*ag*bc*bf*de*eg+af*ag*bc*bf*de*eg+
                  ac*af*bd*bf*de*eg+ac*ag*bd*bf*de*eg+ac*af*bc*bg*de*eg+ad*af*bc*bg*de*eg+ac*ad*bf*bg*de*eg+ac*af*bf*bg*de*eg+
                  ab*af*bc*cd*de*eg+af*ag*bc*cd*de*eg+ab*ac*bf*cd*de*eg+ac*af*bf*cd*de*eg+ab*ag*bf*cd*de*eg+af*ag*bf*cd*de*eg+
                  af*bc*bf*cd*de*eg+ag*bc*bf*cd*de*eg+ab*af*bg*cd*de*eg+ac*af*bg*cd*de*eg+ac*bf*bg*cd*de*eg+af*bf*bg*cd*de*eg+
                  ab*ad*bc*cf*de*eg+ad*af*bc*cf*de*eg+ab*ag*bc*cf*de*eg+af*ag*bc*cf*de*eg+ab*ac*bd*cf*de*eg+ab*af*bd*cf*de*eg+
                  ac*ag*bd*cf*de*eg+af*ag*bd*cf*de*eg+af*bc*bd*cf*de*eg+ag*bc*bd*cf*de*eg+ab*ad*bf*cf*de*eg+ac*ad*bf*cf*de*eg+
                  ab*ag*bf*cf*de*eg+ac*ag*bf*cf*de*eg+ac*bd*bf*cf*de*eg+ag*bd*bf*cf*de*eg+ab*ac*bg*cf*de*eg+ac*ad*bg*cf*de*eg+
                  ab*af*bg*cf*de*eg+ad*af*bg*cf*de*eg+ad*bc*bg*cf*de*eg+af*bc*bg*cf*de*eg+ac*bf*bg*cf*de*eg+ad*bf*bg*cf*de*eg+
                  ab*af*cd*cf*de*eg+ab*ag*cd*cf*de*eg+ab*bf*cd*cf*de*eg+ag*bf*cd*cf*de*eg+ab*bg*cd*cf*de*eg+af*bg*cd*cf*de*eg+
                  ab*af*bc*cg*de*eg+ad*af*bc*cg*de*eg+ab*af*bd*cg*de*eg+ac*af*bd*cg*de*eg+ab*ac*bf*cg*de*eg+ab*ad*bf*cg*de*eg+
                  ac*af*bf*cg*de*eg+ad*af*bf*cg*de*eg+ad*bc*bf*cg*de*eg+af*bc*bf*cg*de*eg+ac*bd*bf*cg*de*eg+af*bd*bf*cg*de*eg+
                  ab*ad*cf*cg*de*eg+ab*af*cf*cg*de*eg+ab*bd*cf*cg*de*eg+af*bd*cf*cg*de*eg+ab*bf*cf*cg*de*eg+ad*bf*cf*cg*de*eg+
                  ac*ae*bc*bd*df*eg+ae*af*bc*bd*df*eg+ac*ag*bc*bd*df*eg+af*ag*bc*bd*df*eg+ac*ad*bc*be*df*eg+ac*af*bc*be*df*eg+
                  ad*ag*bc*be*df*eg+af*ag*bc*be*df*eg+ac*af*bd*be*df*eg+ac*ag*bd*be*df*eg+ac*ae*bc*bf*df*eg+ad*ae*bc*bf*df*eg+
                  ac*ag*bc*bf*df*eg+ad*ag*bc*bf*df*eg+ac*ad*be*bf*df*eg+ac*ag*be*bf*df*eg+ac*ad*bc*bg*df*eg+ad*ae*bc*bg*df*eg+
                  ac*af*bc*bg*df*eg+ae*af*bc*bg*df*eg+ac*ae*bd*bg*df*eg+ac*af*bd*bg*df*eg+ac*ad*bf*bg*df*eg+ac*ae*bf*bg*df*eg+
                  ab*ae*bc*cd*df*eg+ae*af*bc*cd*df*eg+ab*ag*bc*cd*df*eg+af*ag*bc*cd*df*eg+ab*ac*be*cd*df*eg+ab*af*be*cd*df*eg+
                  ac*ag*be*cd*df*eg+af*ag*be*cd*df*eg+af*bc*be*cd*df*eg+ag*bc*be*cd*df*eg+ab*ae*bf*cd*df*eg+ac*ae*bf*cd*df*eg+
                  ab*ag*bf*cd*df*eg+ac*ag*bf*cd*df*eg+ac*be*bf*cd*df*eg+ag*be*bf*cd*df*eg+ab*ac*bg*cd*df*eg+ac*ae*bg*cd*df*eg+
                  ab*af*bg*cd*df*eg+ae*af*bg*cd*df*eg+ae*bc*bg*cd*df*eg+af*bc*bg*cd*df*eg+ac*bf*bg*cd*df*eg+ae*bf*bg*cd*df*eg+
                  ab*ad*bc*ce*df*eg+ab*af*bc*ce*df*eg+ad*ag*bc*ce*df*eg+af*ag*bc*ce*df*eg+ab*ac*bd*ce*df*eg+ac*af*bd*ce*df*eg+
                  ab*ag*bd*ce*df*eg+af*ag*bd*ce*df*eg+af*bc*bd*ce*df*eg+ag*bc*bd*ce*df*eg+ab*ac*bf*ce*df*eg+ac*ad*bf*ce*df*eg+
                  ab*ag*bf*ce*df*eg+ad*ag*bf*ce*df*eg+ad*bc*bf*ce*df*eg+ag*bc*bf*ce*df*eg+ab*ad*bg*ce*df*eg+ac*ad*bg*ce*df*eg+
                  ab*af*bg*ce*df*eg+ac*af*bg*ce*df*eg+ac*bd*bg*ce*df*eg+af*bd*bg*ce*df*eg+ac*bf*bg*ce*df*eg+ad*bf*bg*ce*df*eg+
                  ab*af*cd*ce*df*eg+ab*ag*cd*ce*df*eg+ab*bf*cd*ce*df*eg+ag*bf*cd*ce*df*eg+ab*bg*cd*ce*df*eg+af*bg*cd*ce*df*eg+
                  ab*ae*bc*cf*df*eg+ad*ae*bc*cf*df*eg+ab*ag*bc*cf*df*eg+ad*ag*bc*cf*df*eg+ab*ae*bd*cf*df*eg+ac*ae*bd*cf*df*eg+
                  ab*ag*bd*cf*df*eg+ac*ag*bd*cf*df*eg+ab*ac*be*cf*df*eg+ab*ad*be*cf*df*eg+ac*ag*be*cf*df*eg+ad*ag*be*cf*df*eg+
                  ad*bc*be*cf*df*eg+ag*bc*be*cf*df*eg+ac*bd*be*cf*df*eg+ag*bd*be*cf*df*eg+ab*ac*bg*cf*df*eg+ab*ad*bg*cf*df*eg+
                  ac*ae*bg*cf*df*eg+ad*ae*bg*cf*df*eg+ad*bc*bg*cf*df*eg+ae*bc*bg*cf*df*eg+ac*bd*bg*cf*df*eg+ae*bd*bg*cf*df*eg+
                  ab*ad*ce*cf*df*eg+ab*ag*ce*cf*df*eg+ab*bd*ce*cf*df*eg+ag*bd*ce*cf*df*eg+ab*bg*ce*cf*df*eg+ad*bg*ce*cf*df*eg+
                  ab*ad*bc*cg*df*eg+ad*ae*bc*cg*df*eg+ab*af*bc*cg*df*eg+ae*af*bc*cg*df*eg+ab*ac*bd*cg*df*eg+ab*ae*bd*cg*df*eg+
                  ac*af*bd*cg*df*eg+ae*af*bd*cg*df*eg+ae*bc*bd*cg*df*eg+af*bc*bd*cg*df*eg+ab*ad*be*cg*df*eg+ac*ad*be*cg*df*eg+
                  ab*af*be*cg*df*eg+ac*af*be*cg*df*eg+ac*bd*be*cg*df*eg+af*bd*be*cg*df*eg+ab*ac*bf*cg*df*eg+ac*ad*bf*cg*df*eg+
                  ab*ae*bf*cg*df*eg+ad*ae*bf*cg*df*eg+ad*bc*bf*cg*df*eg+ae*bc*bf*cg*df*eg+ac*be*bf*cg*df*eg+ad*be*bf*cg*df*eg+
                  ab*ae*cd*cg*df*eg+ab*af*cd*cg*df*eg+ab*be*cd*cg*df*eg+af*be*cd*cg*df*eg+ab*bf*cd*cg*df*eg+ae*bf*cd*cg*df*eg+
                  ab*ad*cf*cg*df*eg+ab*ae*cf*cg*df*eg+ab*bd*cf*cg*df*eg+ae*bd*cf*cg*df*eg+ab*be*cf*cg*df*eg+ad*be*cf*cg*df*eg+
                  ab*af*bc*de*df*eg+ac*af*bc*de*df*eg+ab*ag*bc*de*df*eg+ac*ag*bc*de*df*eg+ab*ac*bf*de*df*eg+ac*ag*bf*de*df*eg+
                  ac*bc*bf*de*df*eg+ag*bc*bf*de*df*eg+ab*ac*bg*de*df*eg+ac*af*bg*de*df*eg+ac*bc*bg*de*df*eg+af*bc*bg*de*df*eg+
                  ab*ac*cf*de*df*eg+ab*ag*cf*de*df*eg+ab*bc*cf*de*df*eg+ag*bc*cf*de*df*eg+ab*bg*cf*de*df*eg+ac*bg*cf*de*df*eg+
                  ab*ac*cg*de*df*eg+ab*af*cg*de*df*eg+ab*bc*cg*de*df*eg+af*bc*cg*de*df*eg+ab*bf*cg*de*df*eg+ac*bf*cg*de*df*eg+
                  ac*af*bc*bd*dg*eg+ae*af*bc*bd*dg*eg+ac*af*bc*be*dg*eg+ad*af*bc*be*dg*eg+ac*ad*bc*bf*dg*eg+ac*ae*bc*bf*dg*eg+
                  ad*af*bc*bf*dg*eg+ae*af*bc*bf*dg*eg+ac*ae*bd*bf*dg*eg+ac*af*bd*bf*dg*eg+ac*ad*be*bf*dg*eg+ac*af*be*bf*dg*eg+
                  ab*af*bc*cd*dg*eg+ae*af*bc*cd*dg*eg+ab*af*be*cd*dg*eg+ac*af*be*cd*dg*eg+ab*ac*bf*cd*dg*eg+ab*ae*bf*cd*dg*eg+
                  ac*af*bf*cd*dg*eg+ae*af*bf*cd*dg*eg+ae*bc*bf*cd*dg*eg+af*bc*bf*cd*dg*eg+ac*be*bf*cd*dg*eg+af*be*bf*cd*dg*eg+
                  ab*af*bc*ce*dg*eg+ad*af*bc*ce*dg*eg+ab*af*bd*ce*dg*eg+ac*af*bd*ce*dg*eg+ab*ac*bf*ce*dg*eg+ab*ad*bf*ce*dg*eg+
                  ac*af*bf*ce*dg*eg+ad*af*bf*ce*dg*eg+ad*bc*bf*ce*dg*eg+af*bc*bf*ce*dg*eg+ac*bd*bf*ce*dg*eg+af*bd*bf*ce*dg*eg+
                  ab*ad*bc*cf*dg*eg+ab*ae*bc*cf*dg*eg+ad*af*bc*cf*dg*eg+ae*af*bc*cf*dg*eg+ab*ac*bd*cf*dg*eg+ac*ae*bd*cf*dg*eg+
                  ab*af*bd*cf*dg*eg+ae*af*bd*cf*dg*eg+ae*bc*bd*cf*dg*eg+af*bc*bd*cf*dg*eg+ab*ac*be*cf*dg*eg+ac*ad*be*cf*dg*eg+
                  ab*af*be*cf*dg*eg+ad*af*be*cf*dg*eg+ad*bc*be*cf*dg*eg+af*bc*be*cf*dg*eg+ab*ad*bf*cf*dg*eg+ac*ad*bf*cf*dg*eg+
                  ab*ae*bf*cf*dg*eg+ac*ae*bf*cf*dg*eg+ac*bd*bf*cf*dg*eg+ae*bd*bf*cf*dg*eg+ac*be*bf*cf*dg*eg+ad*be*bf*cf*dg*eg+
                  ab*ae*cd*cf*dg*eg+ab*af*cd*cf*dg*eg+ab*be*cd*cf*dg*eg+af*be*cd*cf*dg*eg+ab*bf*cd*cf*dg*eg+ae*bf*cd*cf*dg*eg+
                  ab*ad*ce*cf*dg*eg+ab*af*ce*cf*dg*eg+ab*bd*ce*cf*dg*eg+af*bd*ce*cf*dg*eg+ab*bf*ce*cf*dg*eg+ad*bf*ce*cf*dg*eg+
                  ab*ae*bc*df*dg*eg+ac*ae*bc*df*dg*eg+ab*af*bc*df*dg*eg+ac*af*bc*df*dg*eg+ab*ac*be*df*dg*eg+ac*af*be*df*dg*eg+
                  ac*bc*be*df*dg*eg+af*bc*be*df*dg*eg+ab*ac*bf*df*dg*eg+ac*ae*bf*df*dg*eg+ac*bc*bf*df*dg*eg+ae*bc*bf*df*dg*eg+
                  ab*ac*ce*df*dg*eg+ab*af*ce*df*dg*eg+ab*bc*ce*df*dg*eg+af*bc*ce*df*dg*eg+ab*bf*ce*df*dg*eg+ac*bf*ce*df*dg*eg+
                  ab*ac*cf*df*dg*eg+ab*ae*cf*df*dg*eg+ab*bc*cf*df*dg*eg+ae*bc*cf*df*dg*eg+ab*be*cf*df*dg*eg+ac*be*cf*df*dg*eg+
                  ac*af*bc*bd*ef*eg+ad*af*bc*bd*ef*eg+ac*ag*bc*bd*ef*eg+ad*ag*bc*bd*ef*eg+ac*ad*bc*bf*ef*eg+ad*ag*bc*bf*ef*eg+
                  ac*ad*bd*bf*ef*eg+ac*ag*bd*bf*ef*eg+ac*ad*bc*bg*ef*eg+ad*af*bc*bg*ef*eg+ac*ad*bd*bg*ef*eg+ac*af*bd*bg*ef*eg+
                  ab*af*bc*cd*ef*eg+ad*af*bc*cd*ef*eg+ab*ag*bc*cd*ef*eg+ad*ag*bc*cd*ef*eg+ab*af*bd*cd*ef*eg+ac*af*bd*cd*ef*eg+
                  ab*ag*bd*cd*ef*eg+ac*ag*bd*cd*ef*eg+ab*ac*bf*cd*ef*eg+ab*ad*bf*cd*ef*eg+ac*ag*bf*cd*ef*eg+ad*ag*bf*cd*ef*eg+
                  ad*bc*bf*cd*ef*eg+ag*bc*bf*cd*ef*eg+ac*bd*bf*cd*ef*eg+ag*bd*bf*cd*ef*eg+ab*ac*bg*cd*ef*eg+ab*ad*bg*cd*ef*eg+
                  ac*af*bg*cd*ef*eg+ad*af*bg*cd*ef*eg+ad*bc*bg*cd*ef*eg+af*bc*bg*cd*ef*eg+ac*bd*bg*cd*ef*eg+af*bd*bg*cd*ef*eg+
                  ab*ad*bc*cf*ef*eg+ad*ag*bc*cf*ef*eg+ab*ac*bd*cf*ef*eg+ac*ad*bd*cf*ef*eg+ab*ag*bd*cf*ef*eg+ad*ag*bd*cf*ef*eg+
                  ad*bc*bd*cf*ef*eg+ag*bc*bd*cf*ef*eg+ab*ad*bg*cf*ef*eg+ac*ad*bg*cf*ef*eg+ac*bd*bg*cf*ef*eg+ad*bd*bg*cf*ef*eg+
                  ab*ad*cd*cf*ef*eg+ab*ag*cd*cf*ef*eg+ab*bd*cd*cf*ef*eg+ag*bd*cd*cf*ef*eg+ab*bg*cd*cf*ef*eg+ad*bg*cd*cf*ef*eg+
                  ab*ad*bc*cg*ef*eg+ad*af*bc*cg*ef*eg+ab*ac*bd*cg*ef*eg+ac*ad*bd*cg*ef*eg+ab*af*bd*cg*ef*eg+ad*af*bd*cg*ef*eg+
                  ad*bc*bd*cg*ef*eg+af*bc*bd*cg*ef*eg+ab*ad*bf*cg*ef*eg+ac*ad*bf*cg*ef*eg+ac*bd*bf*cg*ef*eg+ad*bd*bf*cg*ef*eg+
                  ab*ad*cd*cg*ef*eg+ab*af*cd*cg*ef*eg+ab*bd*cd*cg*ef*eg+af*bd*cd*cg*ef*eg+ab*bf*cd*cg*ef*eg+ad*bf*cd*cg*ef*eg+
                  ab*ad*bc*df*ef*eg+ac*ad*bc*df*ef*eg+ab*ag*bc*df*ef*eg+ac*ag*bc*df*ef*eg+ab*ac*bd*df*ef*eg+ac*ag*bd*df*ef*eg+
                  ac*bc*bd*df*ef*eg+ag*bc*bd*df*ef*eg+ab*ac*bg*df*ef*eg+ac*ad*bg*df*ef*eg+ac*bc*bg*df*ef*eg+ad*bc*bg*df*ef*eg+
                  ab*ac*cd*df*ef*eg+ab*ag*cd*df*ef*eg+ab*bc*cd*df*ef*eg+ag*bc*cd*df*ef*eg+ab*bg*cd*df*ef*eg+ac*bg*cd*df*ef*eg+
                  ab*ac*cg*df*ef*eg+ab*ad*cg*df*ef*eg+ab*bc*cg*df*ef*eg+ad*bc*cg*df*ef*eg+ab*bd*cg*df*ef*eg+ac*bd*cg*df*ef*eg+
                  ab*ad*bc*dg*ef*eg+ac*ad*bc*dg*ef*eg+ab*af*bc*dg*ef*eg+ac*af*bc*dg*ef*eg+ab*ac*bd*dg*ef*eg+ac*af*bd*dg*ef*eg+
                  ac*bc*bd*dg*ef*eg+af*bc*bd*dg*ef*eg+ab*ac*bf*dg*ef*eg+ac*ad*bf*dg*ef*eg+ac*bc*bf*dg*ef*eg+ad*bc*bf*dg*ef*eg+
                  ab*ac*cd*dg*ef*eg+ab*af*cd*dg*ef*eg+ab*bc*cd*dg*ef*eg+af*bc*cd*dg*ef*eg+ab*bf*cd*dg*ef*eg+ac*bf*cd*dg*ef*eg+
                  ab*ac*cf*dg*ef*eg+ab*ad*cf*dg*ef*eg+ab*bc*cf*dg*ef*eg+ad*bc*cf*dg*ef*eg+ab*bd*cf*dg*ef*eg+ac*bd*cf*dg*ef*eg+
                  ad*af*bc*be*cd*fg+ae*af*bc*be*cd*fg+ad*ag*bc*be*cd*fg+ae*ag*bc*be*cd*fg+ac*af*bd*be*cd*fg+ae*af*bd*be*cd*fg+
                  ac*ag*bd*be*cd*fg+ae*ag*bd*be*cd*fg+ad*ae*bc*bf*cd*fg+ae*ag*bc*bf*cd*fg+ac*ae*bd*bf*cd*fg+ae*ag*bd*bf*cd*fg+
                  ac*ae*be*bf*cd*fg+ad*ae*be*bf*cd*fg+ac*ag*be*bf*cd*fg+ad*ag*be*bf*cd*fg+ad*ae*bc*bg*cd*fg+ae*af*bc*bg*cd*fg+
                  ac*ae*bd*bg*cd*fg+ae*af*bd*bg*cd*fg+ac*ae*be*bg*cd*fg+ad*ae*be*bg*cd*fg+ac*af*be*bg*cd*fg+ad*af*be*bg*cd*fg+
                  ad*af*bc*bd*ce*fg+ae*af*bc*bd*ce*fg+ad*ag*bc*bd*ce*fg+ae*ag*bc*bd*ce*fg+ac*af*bd*be*ce*fg+ad*af*bd*be*ce*fg+
                  ac*ag*bd*be*ce*fg+ad*ag*bd*be*ce*fg+ad*ae*bc*bf*ce*fg+ad*ag*bc*bf*ce*fg+ac*ad*bd*bf*ce*fg+ad*ae*bd*bf*ce*fg+
                  ac*ag*bd*bf*ce*fg+ae*ag*bd*bf*ce*fg+ac*ad*be*bf*ce*fg+ad*ag*be*bf*ce*fg+ad*ae*bc*bg*ce*fg+ad*af*bc*bg*ce*fg+
                  ac*ad*bd*bg*ce*fg+ad*ae*bd*bg*ce*fg+ac*af*bd*bg*ce*fg+ae*af*bd*bg*ce*fg+ac*ad*be*bg*ce*fg+ad*af*be*bg*ce*fg+
                  ab*af*bd*cd*ce*fg+ae*af*bd*cd*ce*fg+ab*ag*bd*cd*ce*fg+ae*ag*bd*cd*ce*fg+ab*af*be*cd*ce*fg+ad*af*be*cd*ce*fg+
                  ab*ag*be*cd*ce*fg+ad*ag*be*cd*ce*fg+ab*ad*bf*cd*ce*fg+ab*ae*bf*cd*ce*fg+ad*ag*bf*cd*ce*fg+ae*ag*bf*cd*ce*fg+
                  ae*bd*bf*cd*ce*fg+ag*bd*bf*cd*ce*fg+ad*be*bf*cd*ce*fg+ag*be*bf*cd*ce*fg+ab*ad*bg*cd*ce*fg+ab*ae*bg*cd*ce*fg+
                  ad*af*bg*cd*ce*fg+ae*af*bg*cd*ce*fg+ae*bd*bg*cd*ce*fg+af*bd*bg*cd*ce*fg+ad*be*bg*cd*ce*fg+af*be*bg*cd*ce*fg+
                  ad*ae*bc*bd*cf*fg+ae*ag*bc*bd*cf*fg+ad*ae*bc*be*cf*fg+ad*ag*bc*be*cf*fg+ac*ad*bd*be*cf*fg+ac*ae*bd*be*cf*fg+
                  ad*ag*bd*be*cf*fg+ae*ag*bd*be*cf*fg+ac*ae*bd*bg*cf*fg+ad*ae*bd*bg*cf*fg+ac*ad*be*bg*cf*fg+ad*ae*be*bg*cf*fg+
                  ab*ae*bd*cd*cf*fg+ae*ag*bd*cd*cf*fg+ab*ad*be*cd*cf*fg+ad*ae*be*cd*cf*fg+ab*ag*be*cd*cf*fg+ae*ag*be*cd*cf*fg+
                  ae*bd*be*cd*cf*fg+ag*bd*be*cd*cf*fg+ab*ae*bg*cd*cf*fg+ad*ae*bg*cd*cf*fg+ad*be*bg*cd*cf*fg+ae*be*bg*cd*cf*fg+
                  ab*ae*bd*ce*cf*fg+ad*ae*bd*ce*cf*fg+ab*ag*bd*ce*cf*fg+ad*ag*bd*ce*cf*fg+ab*ad*be*ce*cf*fg+ad*ag*be*ce*cf*fg+
                  ad*bd*be*ce*cf*fg+ag*bd*be*ce*cf*fg+ab*ad*bg*ce*cf*fg+ad*ae*bg*ce*cf*fg+ad*bd*bg*ce*cf*fg+ae*bd*bg*ce*cf*fg+
                  ad*ae*bc*bd*cg*fg+ae*af*bc*bd*cg*fg+ad*ae*bc*be*cg*fg+ad*af*bc*be*cg*fg+ac*ad*bd*be*cg*fg+ac*ae*bd*be*cg*fg+
                  ad*af*bd*be*cg*fg+ae*af*bd*be*cg*fg+ac*ae*bd*bf*cg*fg+ad*ae*bd*bf*cg*fg+ac*ad*be*bf*cg*fg+ad*ae*be*bf*cg*fg+
                  ab*ae*bd*cd*cg*fg+ae*af*bd*cd*cg*fg+ab*ad*be*cd*cg*fg+ad*ae*be*cd*cg*fg+ab*af*be*cd*cg*fg+ae*af*be*cd*cg*fg+
                  ae*bd*be*cd*cg*fg+af*bd*be*cd*cg*fg+ab*ae*bf*cd*cg*fg+ad*ae*bf*cd*cg*fg+ad*be*bf*cd*cg*fg+ae*be*bf*cd*cg*fg+
                  ab*ae*bd*ce*cg*fg+ad*ae*bd*ce*cg*fg+ab*af*bd*ce*cg*fg+ad*af*bd*ce*cg*fg+ab*ad*be*ce*cg*fg+ad*af*be*ce*cg*fg+
                  ad*bd*be*ce*cg*fg+af*bd*be*ce*cg*fg+ab*ad*bf*ce*cg*fg+ad*ae*bf*ce*cg*fg+ad*bd*bf*ce*cg*fg+ae*bd*bf*ce*cg*fg+
                  ac*af*bc*bd*de*fg+ae*af*bc*bd*de*fg+ac*ag*bc*bd*de*fg+ae*ag*bc*bd*de*fg+ac*af*bc*be*de*fg+ad*af*bc*be*de*fg+
                  ac*ag*bc*be*de*fg+ad*ag*bc*be*de*fg+ac*ad*bc*bf*de*fg+ac*ae*bc*bf*de*fg+ad*ag*bc*bf*de*fg+ae*ag*bc*bf*de*fg+
                  ac*ae*bd*bf*de*fg+ac*ag*bd*bf*de*fg+ac*ad*be*bf*de*fg+ac*ag*be*bf*de*fg+ac*ad*bc*bg*de*fg+ac*ae*bc*bg*de*fg+
                  ad*af*bc*bg*de*fg+ae*af*bc*bg*de*fg+ac*ae*bd*bg*de*fg+ac*af*bd*bg*de*fg+ac*ad*be*bg*de*fg+ac*af*be*bg*de*fg+
                  ab*af*bc*cd*de*fg+ae*af*bc*cd*de*fg+ab*ag*bc*cd*de*fg+ae*ag*bc*cd*de*fg+ab*af*be*cd*de*fg+ac*af*be*cd*de*fg+
                  ab*ag*be*cd*de*fg+ac*ag*be*cd*de*fg+ab*ac*bf*cd*de*fg+ab*ae*bf*cd*de*fg+ac*ag*bf*cd*de*fg+ae*ag*bf*cd*de*fg+
                  ae*bc*bf*cd*de*fg+ag*bc*bf*cd*de*fg+ac*be*bf*cd*de*fg+ag*be*bf*cd*de*fg+ab*ac*bg*cd*de*fg+ab*ae*bg*cd*de*fg+
                  ac*af*bg*cd*de*fg+ae*af*bg*cd*de*fg+ae*bc*bg*cd*de*fg+af*bc*bg*cd*de*fg+ac*be*bg*cd*de*fg+af*be*bg*cd*de*fg+
                  ab*af*bc*ce*de*fg+ad*af*bc*ce*de*fg+ab*ag*bc*ce*de*fg+ad*ag*bc*ce*de*fg+ab*af*bd*ce*de*fg+ac*af*bd*ce*de*fg+
                  ab*ag*bd*ce*de*fg+ac*ag*bd*ce*de*fg+ab*ac*bf*ce*de*fg+ab*ad*bf*ce*de*fg+ac*ag*bf*ce*de*fg+ad*ag*bf*ce*de*fg+
                  ad*bc*bf*ce*de*fg+ag*bc*bf*ce*de*fg+ac*bd*bf*ce*de*fg+ag*bd*bf*ce*de*fg+ab*ac*bg*ce*de*fg+ab*ad*bg*ce*de*fg+
                  ac*af*bg*ce*de*fg+ad*af*bg*ce*de*fg+ad*bc*bg*ce*de*fg+af*bc*bg*ce*de*fg+ac*bd*bg*ce*de*fg+af*bd*bg*ce*de*fg+
                  ab*ad*bc*cf*de*fg+ab*ae*bc*cf*de*fg+ad*ag*bc*cf*de*fg+ae*ag*bc*cf*de*fg+ab*ac*bd*cf*de*fg+ac*ae*bd*cf*de*fg+
                  ab*ag*bd*cf*de*fg+ae*ag*bd*cf*de*fg+ae*bc*bd*cf*de*fg+ag*bc*bd*cf*de*fg+ab*ac*be*cf*de*fg+ac*ad*be*cf*de*fg+
                  ab*ag*be*cf*de*fg+ad*ag*be*cf*de*fg+ad*bc*be*cf*de*fg+ag*bc*be*cf*de*fg+ab*ad*bg*cf*de*fg+ac*ad*bg*cf*de*fg+
                  ab*ae*bg*cf*de*fg+ac*ae*bg*cf*de*fg+ac*bd*bg*cf*de*fg+ae*bd*bg*cf*de*fg+ac*be*bg*cf*de*fg+ad*be*bg*cf*de*fg+
                  ab*ae*cd*cf*de*fg+ab*ag*cd*cf*de*fg+ab*be*cd*cf*de*fg+ag*be*cd*cf*de*fg+ab*bg*cd*cf*de*fg+ae*bg*cd*cf*de*fg+
                  ab*ad*ce*cf*de*fg+ab*ag*ce*cf*de*fg+ab*bd*ce*cf*de*fg+ag*bd*ce*cf*de*fg+ab*bg*ce*cf*de*fg+ad*bg*ce*cf*de*fg+
                  ab*ad*bc*cg*de*fg+ab*ae*bc*cg*de*fg+ad*af*bc*cg*de*fg+ae*af*bc*cg*de*fg+ab*ac*bd*cg*de*fg+ac*ae*bd*cg*de*fg+
                  ab*af*bd*cg*de*fg+ae*af*bd*cg*de*fg+ae*bc*bd*cg*de*fg+af*bc*bd*cg*de*fg+ab*ac*be*cg*de*fg+ac*ad*be*cg*de*fg+
                  ab*af*be*cg*de*fg+ad*af*be*cg*de*fg+ad*bc*be*cg*de*fg+af*bc*be*cg*de*fg+ab*ad*bf*cg*de*fg+ac*ad*bf*cg*de*fg+
                  ab*ae*bf*cg*de*fg+ac*ae*bf*cg*de*fg+ac*bd*bf*cg*de*fg+ae*bd*bf*cg*de*fg+ac*be*bf*cg*de*fg+ad*be*bf*cg*de*fg+
                  ab*ae*cd*cg*de*fg+ab*af*cd*cg*de*fg+ab*be*cd*cg*de*fg+af*be*cd*cg*de*fg+ab*bf*cd*cg*de*fg+ae*bf*cd*cg*de*fg+
                  ab*ad*ce*cg*de*fg+ab*af*ce*cg*de*fg+ab*bd*ce*cg*de*fg+af*bd*ce*cg*de*fg+ab*bf*ce*cg*de*fg+ad*bf*ce*cg*de*fg+
                  ac*ae*bc*bd*df*fg+ae*ag*bc*bd*df*fg+ac*ad*bc*be*df*fg+ad*ae*bc*be*df*fg+ac*ag*bc*be*df*fg+ae*ag*bc*be*df*fg+
                  ac*ae*bd*be*df*fg+ac*ag*bd*be*df*fg+ac*ae*bc*bg*df*fg+ad*ae*bc*bg*df*fg+ac*ad*be*bg*df*fg+ac*ae*be*bg*df*fg+
                  ab*ae*bc*cd*df*fg+ae*ag*bc*cd*df*fg+ab*ac*be*cd*df*fg+ac*ae*be*cd*df*fg+ab*ag*be*cd*df*fg+ae*ag*be*cd*df*fg+
                  ae*bc*be*cd*df*fg+ag*bc*be*cd*df*fg+ab*ae*bg*cd*df*fg+ac*ae*bg*cd*df*fg+ac*be*bg*cd*df*fg+ae*be*bg*cd*df*fg+
                  ab*ad*bc*ce*df*fg+ad*ae*bc*ce*df*fg+ab*ag*bc*ce*df*fg+ae*ag*bc*ce*df*fg+ab*ac*bd*ce*df*fg+ab*ae*bd*ce*df*fg+
                  ac*ag*bd*ce*df*fg+ae*ag*bd*ce*df*fg+ae*bc*bd*ce*df*fg+ag*bc*bd*ce*df*fg+ab*ad*be*ce*df*fg+ac*ad*be*ce*df*fg+
                  ab*ag*be*ce*df*fg+ac*ag*be*ce*df*fg+ac*bd*be*ce*df*fg+ag*bd*be*ce*df*fg+ab*ac*bg*ce*df*fg+ac*ad*bg*ce*df*fg+
                  ab*ae*bg*ce*df*fg+ad*ae*bg*ce*df*fg+ad*bc*bg*ce*df*fg+ae*bc*bg*ce*df*fg+ac*be*bg*ce*df*fg+ad*be*bg*ce*df*fg+
                  ab*ae*cd*ce*df*fg+ab*ag*cd*ce*df*fg+ab*be*cd*ce*df*fg+ag*be*cd*ce*df*fg+ab*bg*cd*ce*df*fg+ae*bg*cd*ce*df*fg+
                  ab*ae*bc*cg*df*fg+ad*ae*bc*cg*df*fg+ab*ae*bd*cg*df*fg+ac*ae*bd*cg*df*fg+ab*ac*be*cg*df*fg+ab*ad*be*cg*df*fg+
                  ac*ae*be*cg*df*fg+ad*ae*be*cg*df*fg+ad*bc*be*cg*df*fg+ae*bc*be*cg*df*fg+ac*bd*be*cg*df*fg+ae*bd*be*cg*df*fg+
                  ab*ad*ce*cg*df*fg+ab*ae*ce*cg*df*fg+ab*bd*ce*cg*df*fg+ae*bd*ce*cg*df*fg+ab*be*ce*cg*df*fg+ad*be*ce*cg*df*fg+
                  ab*ae*bc*de*df*fg+ac*ae*bc*de*df*fg+ab*ag*bc*de*df*fg+ac*ag*bc*de*df*fg+ab*ac*be*de*df*fg+ac*ag*be*de*df*fg+
                  ac*bc*be*de*df*fg+ag*bc*be*de*df*fg+ab*ac*bg*de*df*fg+ac*ae*bg*de*df*fg+ac*bc*bg*de*df*fg+ae*bc*bg*de*df*fg+
                  ab*ac*ce*de*df*fg+ab*ag*ce*de*df*fg+ab*bc*ce*de*df*fg+ag*bc*ce*de*df*fg+ab*bg*ce*de*df*fg+ac*bg*ce*de*df*fg+
                  ab*ac*cg*de*df*fg+ab*ae*cg*de*df*fg+ab*bc*cg*de*df*fg+ae*bc*cg*de*df*fg+ab*be*cg*de*df*fg+ac*be*cg*de*df*fg+
                  ac*ae*bc*bd*dg*fg+ae*af*bc*bd*dg*fg+ac*ad*bc*be*dg*fg+ad*ae*bc*be*dg*fg+ac*af*bc*be*dg*fg+ae*af*bc*be*dg*fg+
                  ac*ae*bd*be*dg*fg+ac*af*bd*be*dg*fg+ac*ae*bc*bf*dg*fg+ad*ae*bc*bf*dg*fg+ac*ad*be*bf*dg*fg+ac*ae*be*bf*dg*fg+
                  ab*ae*bc*cd*dg*fg+ae*af*bc*cd*dg*fg+ab*ac*be*cd*dg*fg+ac*ae*be*cd*dg*fg+ab*af*be*cd*dg*fg+ae*af*be*cd*dg*fg+
                  ae*bc*be*cd*dg*fg+af*bc*be*cd*dg*fg+ab*ae*bf*cd*dg*fg+ac*ae*bf*cd*dg*fg+ac*be*bf*cd*dg*fg+ae*be*bf*cd*dg*fg+
                  ab*ad*bc*ce*dg*fg+ad*ae*bc*ce*dg*fg+ab*af*bc*ce*dg*fg+ae*af*bc*ce*dg*fg+ab*ac*bd*ce*dg*fg+ab*ae*bd*ce*dg*fg+
                  ac*af*bd*ce*dg*fg+ae*af*bd*ce*dg*fg+ae*bc*bd*ce*dg*fg+af*bc*bd*ce*dg*fg+ab*ad*be*ce*dg*fg+ac*ad*be*ce*dg*fg+
                  ab*af*be*ce*dg*fg+ac*af*be*ce*dg*fg+ac*bd*be*ce*dg*fg+af*bd*be*ce*dg*fg+ab*ac*bf*ce*dg*fg+ac*ad*bf*ce*dg*fg+
                  ab*ae*bf*ce*dg*fg+ad*ae*bf*ce*dg*fg+ad*bc*bf*ce*dg*fg+ae*bc*bf*ce*dg*fg+ac*be*bf*ce*dg*fg+ad*be*bf*ce*dg*fg+
                  ab*ae*cd*ce*dg*fg+ab*af*cd*ce*dg*fg+ab*be*cd*ce*dg*fg+af*be*cd*ce*dg*fg+ab*bf*cd*ce*dg*fg+ae*bf*cd*ce*dg*fg+
                  ab*ae*bc*cf*dg*fg+ad*ae*bc*cf*dg*fg+ab*ae*bd*cf*dg*fg+ac*ae*bd*cf*dg*fg+ab*ac*be*cf*dg*fg+ab*ad*be*cf*dg*fg+
                  ac*ae*be*cf*dg*fg+ad*ae*be*cf*dg*fg+ad*bc*be*cf*dg*fg+ae*bc*be*cf*dg*fg+ac*bd*be*cf*dg*fg+ae*bd*be*cf*dg*fg+
                  ab*ad*ce*cf*dg*fg+ab*ae*ce*cf*dg*fg+ab*bd*ce*cf*dg*fg+ae*bd*ce*cf*dg*fg+ab*be*ce*cf*dg*fg+ad*be*ce*cf*dg*fg+
                  ab*ae*bc*de*dg*fg+ac*ae*bc*de*dg*fg+ab*af*bc*de*dg*fg+ac*af*bc*de*dg*fg+ab*ac*be*de*dg*fg+ac*af*be*de*dg*fg+
                  ac*bc*be*de*dg*fg+af*bc*be*de*dg*fg+ab*ac*bf*de*dg*fg+ac*ae*bf*de*dg*fg+ac*bc*bf*de*dg*fg+ae*bc*bf*de*dg*fg+
                  ab*ac*ce*de*dg*fg+ab*af*ce*de*dg*fg+ab*bc*ce*de*dg*fg+af*bc*ce*de*dg*fg+ab*bf*ce*de*dg*fg+ac*bf*ce*de*dg*fg+
                  ab*ac*cf*de*dg*fg+ab*ae*cf*de*dg*fg+ab*bc*cf*de*dg*fg+ae*bc*cf*de*dg*fg+ab*be*cf*de*dg*fg+ac*be*cf*de*dg*fg+
                  ac*ae*bc*bd*ef*fg+ad*ae*bc*bd*ef*fg+ac*ag*bc*bd*ef*fg+ad*ag*bc*bd*ef*fg+ac*ad*bc*be*ef*fg+ad*ag*bc*be*ef*fg+
                  ac*ad*bd*be*ef*fg+ac*ag*bd*be*ef*fg+ac*ad*bc*bg*ef*fg+ad*ae*bc*bg*ef*fg+ac*ad*bd*bg*ef*fg+ac*ae*bd*bg*ef*fg+
                  ab*ae*bc*cd*ef*fg+ad*ae*bc*cd*ef*fg+ab*ag*bc*cd*ef*fg+ad*ag*bc*cd*ef*fg+ab*ae*bd*cd*ef*fg+ac*ae*bd*cd*ef*fg+
                  ab*ag*bd*cd*ef*fg+ac*ag*bd*cd*ef*fg+ab*ac*be*cd*ef*fg+ab*ad*be*cd*ef*fg+ac*ag*be*cd*ef*fg+ad*ag*be*cd*ef*fg+
                  ad*bc*be*cd*ef*fg+ag*bc*be*cd*ef*fg+ac*bd*be*cd*ef*fg+ag*bd*be*cd*ef*fg+ab*ac*bg*cd*ef*fg+ab*ad*bg*cd*ef*fg+
                  ac*ae*bg*cd*ef*fg+ad*ae*bg*cd*ef*fg+ad*bc*bg*cd*ef*fg+ae*bc*bg*cd*ef*fg+ac*bd*bg*cd*ef*fg+ae*bd*bg*cd*ef*fg+
                  ab*ad*bc*ce*ef*fg+ad*ag*bc*ce*ef*fg+ab*ac*bd*ce*ef*fg+ac*ad*bd*ce*ef*fg+ab*ag*bd*ce*ef*fg+ad*ag*bd*ce*ef*fg+
                  ad*bc*bd*ce*ef*fg+ag*bc*bd*ce*ef*fg+ab*ad*bg*ce*ef*fg+ac*ad*bg*ce*ef*fg+ac*bd*bg*ce*ef*fg+ad*bd*bg*ce*ef*fg+
                  ab*ad*cd*ce*ef*fg+ab*ag*cd*ce*ef*fg+ab*bd*cd*ce*ef*fg+ag*bd*cd*ce*ef*fg+ab*bg*cd*ce*ef*fg+ad*bg*cd*ce*ef*fg+
                  ab*ad*bc*cg*ef*fg+ad*ae*bc*cg*ef*fg+ab*ac*bd*cg*ef*fg+ac*ad*bd*cg*ef*fg+ab*ae*bd*cg*ef*fg+ad*ae*bd*cg*ef*fg+
                  ad*bc*bd*cg*ef*fg+ae*bc*bd*cg*ef*fg+ab*ad*be*cg*ef*fg+ac*ad*be*cg*ef*fg+ac*bd*be*cg*ef*fg+ad*bd*be*cg*ef*fg+
                  ab*ad*cd*cg*ef*fg+ab*ae*cd*cg*ef*fg+ab*bd*cd*cg*ef*fg+ae*bd*cd*cg*ef*fg+ab*be*cd*cg*ef*fg+ad*be*cd*cg*ef*fg+
                  ab*ad*bc*de*ef*fg+ac*ad*bc*de*ef*fg+ab*ag*bc*de*ef*fg+ac*ag*bc*de*ef*fg+ab*ac*bd*de*ef*fg+ac*ag*bd*de*ef*fg+
                  ac*bc*bd*de*ef*fg+ag*bc*bd*de*ef*fg+ab*ac*bg*de*ef*fg+ac*ad*bg*de*ef*fg+ac*bc*bg*de*ef*fg+ad*bc*bg*de*ef*fg+
                  ab*ac*cd*de*ef*fg+ab*ag*cd*de*ef*fg+ab*bc*cd*de*ef*fg+ag*bc*cd*de*ef*fg+ab*bg*cd*de*ef*fg+ac*bg*cd*de*ef*fg+
                  ab*ac*cg*de*ef*fg+ab*ad*cg*de*ef*fg+ab*bc*cg*de*ef*fg+ad*bc*cg*de*ef*fg+ab*bd*cg*de*ef*fg+ac*bd*cg*de*ef*fg+
                  ab*ad*bc*dg*ef*fg+ac*ad*bc*dg*ef*fg+ab*ae*bc*dg*ef*fg+ac*ae*bc*dg*ef*fg+ab*ac*bd*dg*ef*fg+ac*ae*bd*dg*ef*fg+
                  ac*bc*bd*dg*ef*fg+ae*bc*bd*dg*ef*fg+ab*ac*be*dg*ef*fg+ac*ad*be*dg*ef*fg+ac*bc*be*dg*ef*fg+ad*bc*be*dg*ef*fg+
                  ab*ac*cd*dg*ef*fg+ab*ae*cd*dg*ef*fg+ab*bc*cd*dg*ef*fg+ae*bc*cd*dg*ef*fg+ab*be*cd*dg*ef*fg+ac*be*cd*dg*ef*fg+
                  ab*ac*ce*dg*ef*fg+ab*ad*ce*dg*ef*fg+ab*bc*ce*dg*ef*fg+ad*bc*ce*dg*ef*fg+ab*bd*ce*dg*ef*fg+ac*bd*ce*dg*ef*fg+
                  ac*ae*bc*bd*eg*fg+ad*ae*bc*bd*eg*fg+ac*af*bc*bd*eg*fg+ad*af*bc*bd*eg*fg+ac*ad*bc*be*eg*fg+ad*af*bc*be*eg*fg+
                  ac*ad*bd*be*eg*fg+ac*af*bd*be*eg*fg+ac*ad*bc*bf*eg*fg+ad*ae*bc*bf*eg*fg+ac*ad*bd*bf*eg*fg+ac*ae*bd*bf*eg*fg+
                  ab*ae*bc*cd*eg*fg+ad*ae*bc*cd*eg*fg+ab*af*bc*cd*eg*fg+ad*af*bc*cd*eg*fg+ab*ae*bd*cd*eg*fg+ac*ae*bd*cd*eg*fg+
                  ab*af*bd*cd*eg*fg+ac*af*bd*cd*eg*fg+ab*ac*be*cd*eg*fg+ab*ad*be*cd*eg*fg+ac*af*be*cd*eg*fg+ad*af*be*cd*eg*fg+
                  ad*bc*be*cd*eg*fg+af*bc*be*cd*eg*fg+ac*bd*be*cd*eg*fg+af*bd*be*cd*eg*fg+ab*ac*bf*cd*eg*fg+ab*ad*bf*cd*eg*fg+
                  ac*ae*bf*cd*eg*fg+ad*ae*bf*cd*eg*fg+ad*bc*bf*cd*eg*fg+ae*bc*bf*cd*eg*fg+ac*bd*bf*cd*eg*fg+ae*bd*bf*cd*eg*fg+
                  ab*ad*bc*ce*eg*fg+ad*af*bc*ce*eg*fg+ab*ac*bd*ce*eg*fg+ac*ad*bd*ce*eg*fg+ab*af*bd*ce*eg*fg+ad*af*bd*ce*eg*fg+
                  ad*bc*bd*ce*eg*fg+af*bc*bd*ce*eg*fg+ab*ad*bf*ce*eg*fg+ac*ad*bf*ce*eg*fg+ac*bd*bf*ce*eg*fg+ad*bd*bf*ce*eg*fg+
                  ab*ad*cd*ce*eg*fg+ab*af*cd*ce*eg*fg+ab*bd*cd*ce*eg*fg+af*bd*cd*ce*eg*fg+ab*bf*cd*ce*eg*fg+ad*bf*cd*ce*eg*fg+
                  ab*ad*bc*cf*eg*fg+ad*ae*bc*cf*eg*fg+ab*ac*bd*cf*eg*fg+ac*ad*bd*cf*eg*fg+ab*ae*bd*cf*eg*fg+ad*ae*bd*cf*eg*fg+
                  ad*bc*bd*cf*eg*fg+ae*bc*bd*cf*eg*fg+ab*ad*be*cf*eg*fg+ac*ad*be*cf*eg*fg+ac*bd*be*cf*eg*fg+ad*bd*be*cf*eg*fg+
                  ab*ad*cd*cf*eg*fg+ab*ae*cd*cf*eg*fg+ab*bd*cd*cf*eg*fg+ae*bd*cd*cf*eg*fg+ab*be*cd*cf*eg*fg+ad*be*cd*cf*eg*fg+
                  ab*ad*bc*de*eg*fg+ac*ad*bc*de*eg*fg+ab*af*bc*de*eg*fg+ac*af*bc*de*eg*fg+ab*ac*bd*de*eg*fg+ac*af*bd*de*eg*fg+
                  ac*bc*bd*de*eg*fg+af*bc*bd*de*eg*fg+ab*ac*bf*de*eg*fg+ac*ad*bf*de*eg*fg+ac*bc*bf*de*eg*fg+ad*bc*bf*de*eg*fg+
                  ab*ac*cd*de*eg*fg+ab*af*cd*de*eg*fg+ab*bc*cd*de*eg*fg+af*bc*cd*de*eg*fg+ab*bf*cd*de*eg*fg+ac*bf*cd*de*eg*fg+
                  ab*ac*cf*de*eg*fg+ab*ad*cf*de*eg*fg+ab*bc*cf*de*eg*fg+ad*bc*cf*de*eg*fg+ab*bd*cf*de*eg*fg+ac*bd*cf*de*eg*fg+
                  ab*ad*bc*df*eg*fg+ac*ad*bc*df*eg*fg+ab*ae*bc*df*eg*fg+ac*ae*bc*df*eg*fg+ab*ac*bd*df*eg*fg+ac*ae*bd*df*eg*fg+
                  ac*bc*bd*df*eg*fg+ae*bc*bd*df*eg*fg+ab*ac*be*df*eg*fg+ac*ad*be*df*eg*fg+ac*bc*be*df*eg*fg+ad*bc*be*df*eg*fg+
                  ab*ac*cd*df*eg*fg+ab*ae*cd*df*eg*fg+ab*bc*cd*df*eg*fg+ae*bc*cd*df*eg*fg+ab*be*cd*df*eg*fg+ac*be*cd*df*eg*fg+
                  ab*ac*ce*df*eg*fg+ab*ad*ce*df*eg*fg+ab*bc*ce*df*eg*fg+ad*bc*ce*df*eg*fg+ab*bd*ce*df*eg*fg+ac*bd*ce*df*eg*fg);
    }*/

DEVICE inline OverlapReal gam4(OverlapReal ab,OverlapReal ac,OverlapReal ad,
                OverlapReal bc,OverlapReal bd,
                OverlapReal cd)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        return abcd*(abcd-acbd-adbc)+acbd*(acbd-abcd-adbc)+adbc*(adbc-abcd-acbd);
    }

DEVICE inline OverlapReal gam5(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
                OverlapReal bc,OverlapReal bd,OverlapReal be,
                OverlapReal cd,OverlapReal ce,
                OverlapReal de)
    {
        return
        -2*(ab*bc*cd*de*ae+ac*ce*be*bd*ad+
            ab*bd*cd*ce*ae+ad*de*be*bc*ac+
            ab*be*ce*cd*ad+ae*de*bd*bc*ac+
            ab*bc*ce*de*ad+ac*cd*bd*be*ae+
            ab*bd*de*ce*ac+ad*cd*bc*be*ae+
            ab*be*de*cd*ac+ae*ce*bc*bd*ad)

        +2*(ab*ab*cd*ce*de+
            ac*ac*bd*be*de+
            ad*ad*bc*be*ce+
            ae*ae*bc*bd*cd+
            bc*bc*ad*ae*de+
            bd*bd*ac*ae*ce+
            be*be*ac*ad*cd+
            cd*cd*ab*ae*be+
            ce*ce*ab*ad*bd+
            de*de*ab*ac*bc);
    }

DEVICE inline  OverlapReal gam6(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
                OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
                OverlapReal cd,OverlapReal ce,OverlapReal cf,
                OverlapReal de,OverlapReal df,
                OverlapReal ef)
    {
        return
        -2*(ae*af*bd*bf*cd*ce+ad*af*be*bf*cd*ce+ae*af*bd*be*cd*cf+ad*ae*be*bf*cd*cf+ad*af*bd*be*ce*cf+ad*ae*bd*bf*ce*cf+
            ae*af*bc*bf*cd*de+ac*af*be*bf*cd*de+ad*af*bc*bf*ce*de+ac*af*bd*bf*ce*de+ae*af*bc*bd*cf*de+ad*af*bc*be*cf*de+
            ac*ae*bd*bf*cf*de+ac*ad*be*bf*cf*de+ab*af*be*cd*cf*de+ab*ae*bf*cd*cf*de+ab*af*bd*ce*cf*de+ab*ad*bf*ce*cf*de+
            ae*af*bc*be*cd*df+ac*ae*be*bf*cd*df+ae*af*bc*bd*ce*df+ac*af*bd*be*ce*df+ad*ae*bc*bf*ce*df+ac*ad*be*bf*ce*df+
            ab*af*be*cd*ce*df+ab*ae*bf*cd*ce*df+ad*ae*bc*be*cf*df+ac*ae*bd*be*cf*df+ab*ae*bd*ce*cf*df+ab*ad*be*ce*cf*df+
            ac*af*bc*be*de*df+ac*ae*bc*bf*de*df+ab*af*bc*ce*de*df+ab*ac*bf*ce*de*df+ab*ae*bc*cf*de*df+ab*ac*be*cf*de*df+
            ad*af*bc*be*cd*ef+ac*af*bd*be*cd*ef+ad*ae*bc*bf*cd*ef+ac*ae*bd*bf*cd*ef+ad*af*bc*bd*ce*ef+ac*ad*bd*bf*ce*ef+
            ab*af*bd*cd*ce*ef+ab*ad*bf*cd*ce*ef+ad*ae*bc*bd*cf*ef+ac*ad*bd*be*cf*ef+ab*ae*bd*cd*cf*ef+ab*ad*be*cd*cf*ef+
            ac*af*bc*bd*de*ef+ac*ad*bc*bf*de*ef+ab*af*bc*cd*de*ef+ab*ac*bf*cd*de*ef+ab*ad*bc*cf*de*ef+ab*ac*bd*cf*de*ef+
            ac*ae*bc*bd*df*ef+ac*ad*bc*be*df*ef+ab*ae*bc*cd*df*ef+ab*ac*be*cd*df*ef+ab*ad*bc*ce*df*ef+ab*ac*bd*ce*df*ef)

        +2*(ab*ab*(cd*ce*df*ef+cd*cf*de*ef+ce*cf*de*df)+
            ac*ac*(bd*be*df*ef+bd*bf*de*ef+be*bf*de*df)+
            ad*ad*(bc*be*cf*ef+bc*bf*ce*ef+be*bf*ce*cf)+
            ae*ae*(bc*bd*cf*df+bc*bf*cd*df+bd*bf*cd*cf)+
            af*af*(bc*bd*ce*de+bc*be*cd*de+bd*be*cd*ce)+
            bc*bc*(ad*ae*df*ef+ad*af*de*ef+ae*af*de*df)+
            bd*bd*(ac*ae*cf*ef+ac*af*ce*ef+ae*af*ce*cf)+
            be*be*(ac*ad*cf*df+ac*af*cd*df+ad*af*cd*cf)+
            bf*bf*(ac*ad*ce*de+ac*ae*cd*de+ad*ae*cd*ce)+
            cd*cd*(ab*ae*bf*ef+ab*af*be*ef+ae*af*be*bf)+
            ce*ce*(ab*ad*bf*df+ab*af*bd*df+ad*af*bd*bf)+
            cf*cf*(ab*ad*be*de+ab*ae*bd*de+ad*ae*bd*be)+
            de*de*(ab*ac*bf*cf+ab*af*bc*cf+ac*af*bc*bf)+
            df*df*(ab*ac*be*ce+ab*ae*bc*ce+ac*ae*bc*be)+
            ef*ef*(ab*ac*bd*cd+ab*ad*bc*cd+ac*ad*bc*bd))

        +4*(ab*af*bf*cd*ce*de+ac*af*bd*be*cf*de+ad*ae*bc*bf*cf*de+ad*af*bc*be*ce*df+ac*ae*bd*bf*ce*df+ab*ae*be*cd*cf*df+
            ae*af*bc*bd*cd*ef+ac*ad*be*bf*cd*ef+ab*ad*bd*ce*cf*ef+ab*ac*bc*de*df*ef)

        -1*(af*af*be*be*cd*cd+ae*ae*bf*bf*cd*cd+af*af*bd*bd*ce*ce+ad*ad*bf*bf*ce*ce+ae*ae*bd*bd*cf*cf+ad*ad*be*be*cf*cf+
            af*af*bc*bc*de*de+ac*ac*bf*bf*de*de+ab*ab*cf*cf*de*de+ae*ae*bc*bc*df*df+ac*ac*be*be*df*df+ab*ab*ce*ce*df*df+
            ad*ad*bc*bc*ef*ef+ac*ac*bd*bd*ef*ef+ab*ab*cd*cd*ef*ef);
    }

DEVICE inline OverlapReal beta4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
                 OverlapReal bc,OverlapReal bd,OverlapReal be,
                 OverlapReal cd,OverlapReal ce,
                 OverlapReal de)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        return abcd*(abcd-acbd-adbc)+acbd*(acbd-abcd-adbc)+adbc*(adbc-abcd-acbd)
        +ae*(abcd*(bc+bd-cd)+acbd*(bc+cd-bd)+adbc*(bd+cd-bc)-2*bc*bd*bd)
        +be*(abcd*(ac+ad-cd)+adbc*(ac+cd-ad)+acbd*(ad+cd-ac)-2*ac*ad*cd)
        +ce*(acbd*(ab+ad-bd)+adbc*(ab+bd-ad)+abcd*(ad+bd-ab)-2*ab*ad*bd)
        +de*(adbc*(ab+ac-bc)+acbd*(ab+bc-ac)+abcd*(ac+bc-ab)-2*ab*ac*bc);
    }

DEVICE inline OverlapReal ang3(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,
                OverlapReal bc,OverlapReal bd,OverlapReal be,
                OverlapReal cd,OverlapReal ce,
                OverlapReal de)
    {
        return
        (ab*(cd+ce-de)-(ad-bd)*(ae-be))*(ac+bc-ab)+
        (ac*(bd+be-de)-(ad-cd)*(ae-ce))*(ab+bc-ac)+
        (bc*(ad+ae-de)-(bd-cd)*(be-ce))*(ab+ac-bc)-2*ab*ac*bc;
    }

DEVICE inline OverlapReal ang4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,
                OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,
                OverlapReal cd,OverlapReal ce,OverlapReal cf,
                OverlapReal de,OverlapReal df,
                OverlapReal ef)
    {
        OverlapReal abcd = ab*cd,acbd = ac*bd,adbc = ad*bc;
        OverlapReal abc = ab*ac*bc,abd = ab*ad*bd,acd = ac*ad*cd,bcd = bc*bd*cd;

        return abcd*(abcd-acbd-adbc)+acbd*(acbd-abcd-adbc)+adbc*(adbc-abcd-acbd)

        +(ae+af)*(abcd*(bc+bd-cd)+acbd*(bc+cd-bd)+adbc*(bd+cd-bc)-2*bcd)
        +(be+bf)*(abcd*(ac+ad-cd)+adbc*(ac+cd-ad)+acbd*(ad+cd-ac)-2*acd)
        +(ce+cf)*(acbd*(ab+ad-bd)+adbc*(ab+bd-ad)+abcd*(ad+bd-ab)-2*abd)
        +(de+df)*(adbc*(ab+ac-bc)+acbd*(ab+bc-ac)+abcd*(ac+bc-ab)-2*abc)

        -ef*2*(abcd*(ac+bd+ad+bc-ab-cd)+acbd*(ab+cd+ad+bc-ac-bd)+adbc*(ab+cd+ac+bd-ad-bc)-(abc+abd+acd+bcd))

        -(ae-be)*(af-bf)*((cd+ac+ad)*(cd+bc+bd)-2*(cd*(cd+ab)+ac*bc+ad*bd))
        -(ae-ce)*(af-cf)*((bd+ab+ad)*(bd+bc+cd)-2*(bd*(bd+ac)+ab*bc+ad*cd))
        -(ae-de)*(af-df)*((bc+ab+ac)*(bc+bd+cd)-2*(bc*(bc+ad)+ab*bd+ac*cd))
        -(be-ce)*(bf-cf)*((ad+ab+bd)*(ad+ac+cd)-2*(ad*(ad+bc)+ab*ac+bd*cd))
        -(be-de)*(bf-df)*((ac+ab+bc)*(ac+ad+cd)-2*(ac*(ac+bd)+ab*ad+bc*cd))
        -(ce-de)*(cf-df)*((ab+ac+bc)*(ab+ad+bd)-2*(ab*(ab+cd)+ac*ad+bc*bd));
    }

DEVICE inline OverlapReal ang5(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal af,OverlapReal ag,
                OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal bf,OverlapReal bg,
                OverlapReal cd,OverlapReal ce,OverlapReal cf,OverlapReal cg,
                OverlapReal de,OverlapReal df,OverlapReal dg,
                OverlapReal ef,OverlapReal eg,
                OverlapReal fg)
    {
        return gam5(ab,ac,ad,ae,bc,bd,be,cd,ce,de)

        +(af+ag)*beta4(bc,bd,be,ab,cd,ce,ac,de,ad,ae)
        +(bf+bg)*beta4(ac,ad,ae,ab,cd,ce,bc,de,bd,be)
        +(cf+cg)*beta4(ab,ad,ae,ac,bd,be,bc,de,cd,ce)
        +(df+dg)*beta4(ab,ac,ae,ad,bc,be,bd,ce,cd,de)
        +(ef+eg)*beta4(ab,ac,ad,ae,bc,bd,be,cd,ce,de)

        -fg*vok4(ab,ac,ad,ae,bc,bd,be,cd,ce,de)

        -(af-bf)*(ag-bg)*ang3(cd,ce,ac,bc,de,ad,bd,ae,be,ab)
        -(af-cf)*(ag-cg)*ang3(bd,be,ab,bc,de,ad,cd,ae,ce,ac)
        -(af-df)*(ag-dg)*ang3(bc,be,ab,bd,ce,ac,cd,ae,de,ad)
        -(af-ef)*(ag-eg)*ang3(bc,bd,ab,be,cd,ac,ce,ad,de,ae)
        -(bf-cf)*(bg-cg)*ang3(ad,ae,ab,ac,de,bd,cd,be,ce,bc)
        -(bf-df)*(bg-dg)*ang3(ac,ae,ab,ad,ce,bc,cd,be,de,bd)
        -(bf-ef)*(bg-eg)*ang3(ac,ad,ab,ae,cd,bc,ce,bd,de,be)
        -(cf-df)*(cg-dg)*ang3(ab,ae,ac,ad,be,bc,bd,ce,de,cd)
        -(cf-ef)*(cg-eg)*ang3(ab,ad,ac,ae,bd,bc,be,cd,de,ce)
        -(df-ef)*(dg-eg)*ang3(ab,ac,ad,ae,bc,bd,be,cd,ce,de);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE inline OverlapReal t2(OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        OverlapReal ab2 = ab*ab;
        OverlapReal ac2 = ac*ac;
        OverlapReal bc2 = bc*bc;

        return sqrt(4*(ab2*ac2+ab2*bc2+ac2*bc2)-(ab2+ac2+bc2)*(ab2+ac2+bc2))/4;
    }

DEVICE inline OverlapReal cos2(OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        return (ab*ab+ac*ac-bc*bc)/(2*ab*ac);
    }

DEVICE inline OverlapReal t3(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        OverlapReal ab2 = ab*ab;
        OverlapReal ac2 = ac*ac;
        OverlapReal ad2 = ad*ad;
        OverlapReal bc2 = bc*bc;
        OverlapReal bd2 = bd*bd;
        OverlapReal cd2 = cd*cd;

        OverlapReal abTcd = ab2*cd2,abPcd = ab2+cd2;
        OverlapReal acTbd = ac2*bd2,acPbd = ac2+bd2;
        OverlapReal adTbc = ad2*bc2,adPbc = ad2+bc2;

        return sqrt((abTcd+acTbd+adTbc)*(abPcd+acPbd+adPbc)
                    -2*(abTcd*abPcd+acTbd*acPbd+adTbc*adPbc)
                    -(ab2*ac2*bc2+ab2*ad2*bd2+ac2*ad2*cd2+bc2*bd2*cd2))/12;
    }

DEVICE inline OverlapReal cos3(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        OverlapReal ab2 = ab*ab;
        OverlapReal ac2 = ac*ac;
        OverlapReal ad2 = ad*ad;
        OverlapReal bc2 = bc*bc;
        OverlapReal bd2 = bd*bd;
        OverlapReal cd2 = cd*cd;

        return ((ab2+ac2+bc2)*(ab2+ad2+bd2)-2*(ab2*(ab2+cd2)+ac2*ad2+bc2*bd2))/(16*t2(ab,ac,bc)*t2(ab,ad,bd));
    }

DEVICE inline bool flag3(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        OverlapReal ab2 = ab*ab;
        OverlapReal ac2 = ac*ac;
        OverlapReal ad2 = ad*ad;
        OverlapReal bc2 = bc*bc;
        OverlapReal bd2 = bd*bd;
        OverlapReal cd2 = cd*cd;

        return (ab2+ac2+bc2)*(ab2+ad2+bd2)-2*(ab2*(ab2+cd2)+ac2*ad2+bc2*bd2) >= 16*t2(ab,ac,bc)*t2(ab,ad,bd);
    }

DEVICE inline bool flag4(OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal ae,OverlapReal bc,OverlapReal bd,OverlapReal be,OverlapReal cd,OverlapReal ce,OverlapReal de)
    {
        OverlapReal ab2 = ab*ab;
        OverlapReal ac2 = ac*ac;
        OverlapReal ad2 = ad*ad;
        OverlapReal ae2 = ae*ae;
        OverlapReal bc2 = bc*bc;
        OverlapReal bd2 = bd*bd;
        OverlapReal be2 = be*be;
        OverlapReal cd2 = cd*cd;
        OverlapReal ce2 = ce*ce;
        OverlapReal de2 = de*de;

        return (ab2+ac2+bc2)*(ad2*be2+ae2*bd2+ad2*ce2+ae2*cd2+bd2*ce2+be2*cd2)
        +(ab2*ac2+ab2*bc2+ac2*bc2)*(ad2+ae2+bd2+be2+cd2+ce2)
        -2*ab2*ac2*bc2+de2*(ab2*ab2+ac2*ac2+bc2*bc2)
        -2*de2*(ab2*ac2+ab2*bc2+ac2*bc2)
        -2*ab2*(ad2*be2+ae2*bd2+cd2*ce2)
        -2*ac2*(ad2*ce2+ae2*cd2+bd2*be2)
        -2*bc2*(bd2*ce2+be2*cd2+ad2*ae2)
        -(ab2*ab2+ac2*bc2)*(cd2+ce2)
        -(ac2*ac2+ab2*bc2)*(bd2+be2)
        -(bc2*bc2+ab2*ac2)*(ad2+ae2) >= 288*t3(ab,ac,ad,bc,bd,cd)*t3(ab,ac,ae,bc,be,ce);
    }

DEVICE inline  OverlapReal wol2(OverlapReal ar,OverlapReal br,OverlapReal ab);
DEVICE inline    OverlapReal wol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc);
DEVICE inline    OverlapReal wol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd);
DEVICE inline    OverlapReal vol2(OverlapReal ar,OverlapReal br,OverlapReal ab);
DEVICE inline    OverlapReal vol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc);
DEVICE inline    OverlapReal vol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd);
DEVICE inline    OverlapReal xol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc);
DEVICE inline    OverlapReal xol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd);
DEVICE inline    OverlapReal yol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd);

DEVICE inline    OverlapReal uol1(OverlapReal ar)
    {
        return M_PI*4/3*ar*ar*ar;
    }

DEVICE inline    OverlapReal uol2(OverlapReal ar,OverlapReal br,OverlapReal ab)
    {
        if(((ar > 0) || (br > 0)) && (ar+br <= ab)) return 0;

        if(((ar > 0) || (br < 0)) && (ar-br <= -ab)) return uol1(ar);
        if(((br > 0) || (ar < 0)) && (br-ar <= -ab)) return uol1(br);

        return vol2(ar,br,ab);
    }

DEVICE inline  OverlapReal vol2(OverlapReal ar,OverlapReal br,OverlapReal ab)
    {
        if(((ar < 0) || (br < 0)) && (-ar-br <= ab)) return uol1(ar)+uol1(br);

        return wol2(ar,br,ab);
    }

 DEVICE inline    OverlapReal wol2(OverlapReal ar,OverlapReal br,OverlapReal ab)
    {
        return
        +uol1(ar)*(1-cos2(ar,ab,br))/2
        +uol1(br)*(1-cos2(br,ab,ar))/2
        -M_PI*4/3*t2(ar,br,ab)*t2(ar,br,ab)/ab;
    }

 DEVICE inline    OverlapReal uol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        if(((ar > 0) || (br > 0)) && (ar+br <= ab)) return 0;
        if(((ar > 0) || (cr > 0)) && (ar+cr <= ac)) return 0;
        if(((br > 0) || (cr > 0)) && (br+cr <= bc)) return 0;

        bool Abr = ((ar > 0) || (br < 0)) && (ar-br <= -ab);
        bool Acr = ((ar > 0) || (cr < 0)) && (ar-cr <= -ac);
        if(Abr && Acr) return uol1(ar);

        bool Bar = ((br > 0) || (ar < 0)) && (br-ar <= -ab);
        bool Bcr = ((br > 0) || (cr < 0)) && (br-cr <= -bc);
        if(Bar && Bcr) return uol1(br);

        bool Car = ((cr > 0) || (ar < 0)) && (cr-ar <= -ac);
        bool Cbr = ((cr > 0) || (br < 0)) && (cr-br <= -bc);
        if(Car && Cbr) return uol1(cr);

        if(Bar || Car) return vol2(br,cr,bc);
        if(Abr || Cbr) return vol2(ar,cr,ac);
        if(Acr || Bcr) return vol2(ar,br,ab);

        return vol3(ar,br,cr,ab,ac,bc);
    }

 DEVICE inline    OverlapReal vol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        bool abR = ((ar < 0) || (br < 0)) && (-ar-br <= ab);
        bool acR = ((ar < 0) || (cr < 0)) && (-ar-cr <= ac);
        bool bcR = ((br < 0) || (cr < 0)) && (-br-cr <= bc);

        if(abR && acR && bcR) return uol1(ar)+uol1(br)+uol1(cr);

        if(abR && acR) return uol1(ar)+wol2(br,cr,bc);
        if(abR && bcR) return uol1(br)+wol2(ar,cr,ac);
        if(acR && bcR) return uol1(cr)+wol2(ar,br,ab);

        if(bcR) return wol2(ar,br,ab)+wol2(ar,cr,ac)-uol1(ar);
        if(acR) return wol2(ar,br,ab)+wol2(br,cr,bc)-uol1(br);
        if(abR) return wol2(ar,cr,ac)+wol2(br,cr,bc)-uol1(cr);

        return wol3(ar,br,cr,ab,ac,bc);
    }

 DEVICE inline    OverlapReal wol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        bool abf = flag3(ab,ac,ar,bc,br,cr);
        bool acf = flag3(ac,ab,ar,bc,cr,br);
        bool bcf = flag3(bc,ab,br,ac,cr,ar);

        if(abf && acf && bcf) return 0;

        if(abf && acf) return wol2(br,cr,bc);
        if(abf && bcf) return wol2(ar,cr,ac);
        if(acf && bcf) return wol2(ar,br,ab);

        if(bcf) return wol2(ar,br,ab)+wol2(ar,cr,ac)-uol1(ar);
        if(acf) return wol2(ar,br,ab)+wol2(br,cr,bc)-uol1(br);
        if(abf) return wol2(ar,cr,ac)+wol2(br,cr,bc)-uol1(cr);

        return xol3(ar,br,cr,ab,ac,bc);
    }

DEVICE inline   OverlapReal sign(OverlapReal val)
        {
        OverlapReal s(1.0);
        s = copysignf(s,val);
        return s;
        }

DEVICE inline    OverlapReal xol3(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal ab,OverlapReal ac,OverlapReal bc)
    {
        OverlapReal tar = acos(cos3(ar,ab,ac,br,cr,bc)*sign(bc));
        OverlapReal tbr = acos(cos3(br,ab,bc,ar,cr,ac)*sign(ac));
        OverlapReal tcr = acos(cos3(cr,ac,bc,ar,br,ab)*sign(ab));
        OverlapReal tbc = acos(cos3(bc,ab,br,ac,cr,ar)*sign(ar));
        OverlapReal tac = acos(cos3(ac,ab,ar,bc,cr,br)*sign(br));
        OverlapReal tab = acos(cos3(ab,ac,ar,bc,br,cr)*sign(cr));

        return
        +2*t3(ar,br,cr,ab,ac,bc)*sign(ar)*sign(br)*sign(cr)
        +(wol2(ar,br,ab)*tab+
          wol2(ar,cr,ac)*tac+
          wol2(br,cr,bc)*tbc)/M_PI
        -(uol1(ar)*(tab+tac+tar-M_PI)+
          uol1(br)*(tab+tbc+tbr-M_PI)+
          uol1(cr)*(tac+tbc+tcr-M_PI))/(2*M_PI);
    }

DEVICE inline    OverlapReal uol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        if(((ar > 0) || (br > 0)) && (ar+br <= ab)) return 0;
        if(((ar > 0) || (cr > 0)) && (ar+cr <= ac)) return 0;
        if(((ar > 0) || (dr > 0)) && (ar+dr <= ad)) return 0;
        if(((br > 0) || (cr > 0)) && (br+cr <= bc)) return 0;
        if(((br > 0) || (dr > 0)) && (br+dr <= bd)) return 0;
        if(((cr > 0) || (dr > 0)) && (cr+dr <= cd)) return 0;

        bool Abr = ((ar > 0) || (br < 0)) && (ar-br <= -ab);
        bool Acr = ((ar > 0) || (cr < 0)) && (ar-cr <= -ac);
        bool Adr = ((ar > 0) || (dr < 0)) && (ar-dr <= -ad);
        if(Abr && Acr && Adr) return uol1(ar);

        bool Bar = ((br > 0) || (ar < 0)) && (br-ar <= -ab);
        bool Bcr = ((br > 0) || (cr < 0)) && (br-cr <= -bc);
        bool Bdr = ((br > 0) || (dr < 0)) && (br-dr <= -bd);
        if(Bar && Bcr && Bdr) return uol1(br);

        bool Car = ((cr > 0) || (ar < 0)) && (cr-ar <= -ac);
        bool Cbr = ((cr > 0) || (br < 0)) && (cr-br <= -bc);
        bool Cdr = ((cr > 0) || (dr < 0)) && (cr-dr <= -cd);
        if(Car && Cbr && Cdr) return uol1(cr);

        bool Dar = ((dr > 0) || (ar < 0)) && (dr-ar <= -ad);
        bool Dbr = ((dr > 0) || (br < 0)) && (dr-br <= -bd);
        bool Dcr = ((dr > 0) || (cr < 0)) && (dr-cr <= -cd);
        if(Dar && Dbr && Dcr) return uol1(dr);

        if((Acr || Bcr) && (Adr || Bdr)) return vol2(ar,br,ab);
        if((Abr || Cbr) && (Adr || Cdr)) return vol2(ar,cr,ac);
        if((Abr || Dbr) && (Acr || Dcr)) return vol2(ar,dr,ad);
        if((Bar || Car) && (Bdr || Cdr)) return vol2(br,cr,bc);
        if((Bar || Dar) && (Bcr || Dcr)) return vol2(br,dr,bd);
        if((Car || Dar) && (Cbr || Dbr)) return vol2(cr,dr,cd);

        if(Bar || Car || Dar) return vol3(br,cr,dr,bc,bd,cd);
        if(Abr || Cbr || Dbr) return vol3(ar,cr,dr,ac,ad,cd);
        if(Acr || Bcr || Dcr) return vol3(ar,br,dr,ab,ad,bd);
        if(Adr || Bdr || Cdr) return vol3(ar,br,cr,ab,ac,bc);

        return vol4(ar,br,cr,dr,ab,ac,ad,bc,bd,cd);
    }

DEVICE inline     OverlapReal vol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        bool abR = ((ar < 0) || (br < 0)) && (-ar-br <= ab);
        bool acR = ((ar < 0) || (cr < 0)) && (-ar-cr <= ac);
        bool adR = ((ar < 0) || (dr < 0)) && (-ar-dr <= ad);
        bool bcR = ((br < 0) || (cr < 0)) && (-br-cr <= bc);
        bool bdR = ((br < 0) || (dr < 0)) && (-br-dr <= bd);
        bool cdR = ((cr < 0) || (dr < 0)) && (-cr-dr <= cd);

        if(abR && acR && adR && bcR && bdR && cdR) return uol1(ar)+uol1(br)+uol1(cr)+uol1(dr);

        if(!abR && acR && adR && bcR && bdR && cdR) return wol2(ar,br,ab)+uol1(cr)+uol1(dr);
        if(abR && !acR && adR && bcR && bdR && cdR) return wol2(ar,cr,ac)+uol1(br)+uol1(dr);
        if(abR && acR && !adR && bcR && bdR && cdR) return wol2(ar,dr,ad)+uol1(br)+uol1(cr);
        if(abR && acR && adR && !bcR && bdR && cdR) return wol2(br,cr,bc)+uol1(ar)+uol1(dr);
        if(abR && acR && adR && bcR && !bdR && cdR) return wol2(br,dr,bd)+uol1(ar)+uol1(cr);
        if(abR && acR && adR && bcR && bdR && !cdR) return wol2(cr,dr,cd)+uol1(ar)+uol1(br);

        if(!abR && acR && adR && bcR && bdR && !cdR) return wol2(ar,br,ab)+wol2(cr,dr,cd);
        if(abR && !acR && adR && bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol2(br,dr,bd);
        if(abR && acR && !adR && !bcR && bdR && cdR) return wol2(ar,dr,ad)+wol2(br,cr,bc);

        if(!abR && !acR && adR && bcR && bdR && cdR) return wol2(ar,br,ab)+wol2(ar,cr,ac)-uol1(ar);
        if(!abR && acR && !adR && bcR && bdR && cdR) return wol2(ar,br,ab)+wol2(ar,dr,ad)-uol1(ar);
        if(abR && !acR && !adR && bcR && bdR && cdR) return wol2(ar,cr,ac)+wol2(ar,dr,ad)-uol1(ar);
        if(!abR && acR && adR && !bcR && bdR && cdR) return wol2(ar,br,ab)+wol2(br,cr,bc)-uol1(br);
        if(!abR && acR && adR && bcR && !bdR && cdR) return wol2(ar,br,ab)+wol2(br,dr,bd)-uol1(br);
        if(abR && acR && adR && !bcR && !bdR && cdR) return wol2(br,cr,bc)+wol2(br,dr,bd)-uol1(br);
        if(abR && !acR && adR && !bcR && bdR && cdR) return wol2(ar,cr,ac)+wol2(br,cr,bc)-uol1(cr);
        if(abR && !acR && adR && bcR && bdR && !cdR) return wol2(ar,cr,ac)+wol2(cr,dr,cd)-uol1(cr);
        if(abR && acR && adR && !bcR && bdR && !cdR) return wol2(br,cr,bc)+wol2(cr,dr,cd)-uol1(cr);
        if(abR && acR && !adR && bcR && !bdR && cdR) return wol2(ar,dr,ad)+wol2(br,dr,bd)-uol1(dr);
        if(abR && acR && !adR && bcR && bdR && !cdR) return wol2(ar,dr,ad)+wol2(cr,dr,cd)-uol1(dr);
        if(abR && acR && adR && bcR && !bdR && !cdR) return wol2(br,dr,bd)+wol2(cr,dr,cd)-uol1(dr);

        if(!abR && !acR && !adR && bcR && bdR && cdR) return wol2(ar,br,ab)+wol2(ar,cr,ac)+wol2(ar,dr,ad)-2*uol1(ar);
        if(!abR && acR && adR && !bcR && !bdR && cdR) return wol2(ar,br,ab)+wol2(br,cr,bc)+wol2(br,dr,bd)-2*uol1(br);
        if(abR && !acR && adR && !bcR && bdR && !cdR) return wol2(ar,cr,ac)+wol2(br,cr,bc)+wol2(cr,dr,cd)-2*uol1(cr);
        if(abR && acR && !adR && bcR && !bdR && !cdR) return wol2(ar,dr,ad)+wol2(br,dr,bd)+wol2(cr,dr,cd)-2*uol1(dr);

        if(abR && acR && adR && !bcR && !bdR && !cdR) return uol1(ar)+wol3(br,cr,dr,bc,bd,cd);
        if(abR && !acR && !adR && bcR && bdR && !cdR) return uol1(br)+wol3(ar,cr,dr,ac,ad,cd);
        if(!abR && acR && !adR && bcR && !bdR && cdR) return uol1(cr)+wol3(ar,br,dr,ab,ad,bd);
        if(!abR && !acR && adR && !bcR && bdR && cdR) return uol1(dr)+wol3(ar,br,cr,ab,ac,bc);

        if(!abR && !acR && adR && bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol2(br,dr,bd)+wol2(ar,br,ab)-uol1(ar)-uol1(br);
        if(!abR && acR && !adR && !bcR && bdR && cdR) return wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(ar,br,ab)-uol1(ar)-uol1(br);
        if(!abR && !acR && adR && bcR && bdR && !cdR) return wol2(ar,br,ab)+wol2(cr,dr,cd)+wol2(ar,cr,ac)-uol1(ar)-uol1(cr);
        if(abR && !acR && !adR && !bcR && bdR && cdR) return wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(ar,cr,ac)-uol1(ar)-uol1(cr);
        if(!abR && acR && !adR && bcR && bdR && !cdR) return wol2(ar,br,ab)+wol2(cr,dr,cd)+wol2(ar,dr,ad)-uol1(ar)-uol1(dr);
        if(abR && !acR && !adR && bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol2(br,dr,bd)+wol2(ar,dr,ad)-uol1(ar)-uol1(dr);
        if(!abR && acR && adR && !bcR && bdR && !cdR) return wol2(ar,br,ab)+wol2(cr,dr,cd)+wol2(br,cr,bc)-uol1(br)-uol1(cr);
        if(abR && !acR && adR && !bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol2(br,dr,bd)+wol2(br,cr,bc)-uol1(br)-uol1(cr);
        if(!abR && acR && adR && bcR && !bdR && !cdR) return wol2(ar,br,ab)+wol2(cr,dr,cd)+wol2(br,dr,bd)-uol1(br)-uol1(dr);
        if(abR && acR && !adR && !bcR && !bdR && cdR) return wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(br,dr,bd)-uol1(br)-uol1(dr);
        if(abR && !acR && adR && bcR && !bdR && !cdR) return wol2(ar,cr,ac)+wol2(br,dr,bd)+wol2(cr,dr,cd)-uol1(cr)-uol1(dr);
        if(abR && acR && !adR && !bcR && bdR && !cdR) return wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(cr,dr,cd)-uol1(cr)-uol1(dr);

        if(abR && !acR && !adR && !bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(br,dr,bd)-uol1(ar)-uol1(br)-uol1(cr)-uol1(dr);
        if(!abR && acR && !adR && !bcR && bdR && !cdR) return wol2(ar,br,ab)+wol2(ar,dr,ad)+wol2(br,cr,bc)+wol2(cr,dr,cd)-uol1(ar)-uol1(br)-uol1(cr)-uol1(dr);
        if(!abR && !acR && adR && bcR && !bdR && !cdR) return wol2(ar,br,ab)+wol2(ar,cr,ac)+wol2(br,dr,bd)+wol2(cr,dr,cd)-uol1(ar)-uol1(br)-uol1(cr)-uol1(dr);

        if(!abR && !acR && !adR && bcR && bdR && !cdR) return wol2(ar,br,ab)+wol3(ar,cr,dr,ac,ad,cd)-uol1(ar);
        if(!abR && !acR && !adR && bcR && !bdR && cdR) return wol2(ar,cr,ac)+wol3(ar,br,dr,ab,ad,bd)-uol1(ar);
        if(!abR && !acR && !adR && !bcR && bdR && cdR) return wol2(ar,dr,ad)+wol3(ar,br,cr,ab,ac,bc)-uol1(ar);
        if(!abR && acR && adR && !bcR && !bdR && !cdR) return wol2(ar,br,ab)+wol3(br,cr,dr,bc,bd,cd)-uol1(br);
        if(!abR && acR && !adR && !bcR && !bdR && cdR) return wol2(br,cr,bc)+wol3(ar,br,dr,ab,ad,bd)-uol1(br);
        if(!abR && !acR && adR && !bcR && !bdR && cdR) return wol2(br,dr,bd)+wol3(ar,br,cr,ab,ac,bc)-uol1(br);
        if(abR && !acR && adR && !bcR && !bdR && !cdR) return wol2(ar,cr,ac)+wol3(br,cr,dr,bc,bd,cd)-uol1(cr);
        if(abR && !acR && !adR && !bcR && bdR && !cdR) return wol2(br,cr,bc)+wol3(ar,cr,dr,ac,ad,cd)-uol1(cr);
        if(!abR && !acR && adR && !bcR && bdR && !cdR) return wol2(cr,dr,cd)+wol3(ar,br,cr,ab,ac,bc)-uol1(cr);
        if(abR && acR && !adR && !bcR && !bdR && !cdR) return wol2(ar,dr,ad)+wol3(br,cr,dr,bc,bd,cd)-uol1(dr);
        if(abR && !acR && !adR && bcR && !bdR && !cdR) return wol2(br,dr,bd)+wol3(ar,cr,dr,ac,ad,cd)-uol1(dr);
        if(!abR && acR && !adR && bcR && !bdR && !cdR) return wol2(cr,dr,cd)+wol3(ar,br,dr,ab,ad,bd)-uol1(dr);

        if(abR && !acR && !adR && !bcR && !bdR && !cdR) return wol3(ar,cr,dr,ac,ad,cd)+wol3(br,cr,dr,bc,bd,cd)-wol2(cr,dr,cd);
        if(!abR && acR && !adR && !bcR && !bdR && !cdR) return wol3(ar,br,dr,ab,ad,bd)+wol3(br,cr,dr,bc,bd,cd)-wol2(br,dr,bd);
        if(!abR && !acR && adR && !bcR && !bdR && !cdR) return wol3(ar,br,cr,ab,ac,bc)+wol3(br,cr,dr,bc,bd,cd)-wol2(br,cr,bc);
        if(!abR && !acR && !adR && bcR && !bdR && !cdR) return wol3(ar,br,dr,ab,ad,bd)+wol3(ar,cr,dr,ac,ad,cd)-wol2(ar,dr,ad);
        if(!abR && !acR && !adR && !bcR && bdR && !cdR) return wol3(ar,br,cr,ab,ac,bc)+wol3(ar,cr,dr,ac,ad,cd)-wol2(ar,cr,ac);
        if(!abR && !acR && !adR && !bcR && !bdR && cdR) return wol3(ar,br,cr,ab,ac,bc)+wol3(ar,br,dr,ab,ad,bd)-wol2(ar,br,ab);

        return wol4(ar,br,cr,dr,ab,ac,ad,bc,bd,cd);
    }

DEVICE inline     OverlapReal wol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        bool bcag = flag3(bc,bd,br,cd,cr,dr);
        bool bdag = flag3(bd,bc,br,cd,dr,cr);
        bool cdag = flag3(cd,bc,cr,bd,dr,br);
        if(bcag && bdag && cdag) return 0;

        bool acbg = flag3(ac,ad,ar,cd,cr,dr);
        bool adbg = flag3(ad,ac,ar,cd,dr,cr);
        bool cdbg = flag3(cd,ac,cr,ad,dr,ar);
        if(acbg && adbg && cdbg) return 0;

        bool abcg = flag3(ab,ad,ar,bd,br,dr);
        bool adcg = flag3(ad,ab,ar,bd,dr,br);
        bool bdcg = flag3(bd,ab,br,ad,dr,ar);
        if(abcg && adcg && bdcg) return 0;

        bool abdg = flag3(ab,ac,ar,bc,br,cr);
        bool acdg = flag3(ac,ab,ar,bc,cr,br);
        bool bcdg = flag3(bc,ab,br,ac,cr,ar);
        if(abdg && acdg && bcdg) return 0;

        if(adcg && bdcg && acdg && bcdg) return wol2(ar,br,ab);
        if(adbg && cdbg && abdg && bcdg) return wol2(ar,cr,ac);
        if(acbg && cdbg && abcg && bdcg) return wol2(ar,dr,ad);
        if(bdag && cdag && abdg && acdg) return wol2(br,cr,bc);
        if(bcag && cdag && abcg && adcg) return wol2(br,dr,bd);
        if(bcag && bdag && acbg && adbg) return wol2(cr,dr,cd);

        if(bdag && cdag && bcdg) return wol2(ar,br,ab)+wol2(ar,cr,ac)-uol1(ar);
        if(bcag && cdag && bdcg) return wol2(ar,br,ab)+wol2(ar,dr,ad)-uol1(ar);
        if(bcag && bdag && cdbg) return wol2(ar,cr,ac)+wol2(ar,dr,ad)-uol1(ar);
        if(adbg && cdbg && acdg) return wol2(ar,br,ab)+wol2(br,cr,bc)-uol1(br);
        if(acbg && cdbg && adcg) return wol2(ar,br,ab)+wol2(br,dr,bd)-uol1(br);
        if(acbg && adbg && cdag) return wol2(br,cr,bc)+wol2(br,dr,bd)-uol1(br);
        if(adcg && bdcg && abdg) return wol2(ar,cr,ac)+wol2(br,cr,bc)-uol1(cr);
        if(abcg && bdcg && adbg) return wol2(ar,cr,ac)+wol2(cr,dr,cd)-uol1(cr);
        if(abcg && adcg && bdag) return wol2(br,cr,bc)+wol2(cr,dr,cd)-uol1(cr);
        if(acdg && bcdg && abcg) return wol2(ar,dr,ad)+wol2(br,dr,bd)-uol1(dr);
        if(abdg && bcdg && acbg) return wol2(ar,dr,ad)+wol2(cr,dr,cd)-uol1(dr);
        if(abdg && acdg && bcag) return wol2(br,dr,bd)+wol2(cr,dr,cd)-uol1(dr);

        if(bdag && cdag) return xol3(ar,br,cr,ab,ac,bc);
        if(bcag && cdag) return xol3(ar,br,dr,ab,ad,bd);
        if(bcag && bdag) return xol3(ar,cr,dr,ac,ad,cd);
        if(adbg && cdbg) return xol3(ar,br,cr,ab,ac,bc);
        if(acbg && cdbg) return xol3(ar,br,dr,ab,ad,bd);
        if(acbg && adbg) return xol3(br,cr,dr,bc,bd,cd);
        if(adcg && bdcg) return xol3(ar,br,cr,ab,ac,bc);
        if(abcg && bdcg) return xol3(ar,cr,dr,ac,ad,cd);
        if(abcg && adcg) return xol3(br,cr,dr,bc,bd,cd);
        if(acdg && bcdg) return xol3(ar,br,dr,ab,ad,bd);
        if(abdg && bcdg) return xol3(ar,cr,dr,ac,ad,cd);
        if(abdg && acdg) return xol3(br,cr,dr,bc,bd,cd);

        if(cdag || cdbg) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,br,dr,ab,ad,bd)-wol2(ar,br,ab);
        if(bdag || bdcg) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,cr,dr,ac,ad,cd)-wol2(ar,cr,ac);
        if(bcag || bcdg) return xol3(ar,br,dr,ab,ad,bd)+xol3(ar,cr,dr,ac,ad,cd)-wol2(ar,dr,ad);
        if(adbg || adcg) return xol3(ar,br,cr,ab,ac,bc)+xol3(br,cr,dr,bc,bd,cd)-wol2(br,cr,bc);
        if(acbg || acdg) return xol3(ar,br,dr,ab,ad,bd)+xol3(br,cr,dr,bc,bd,cd)-wol2(br,dr,bd);
        if(abcg || abdg) return xol3(ar,cr,dr,ac,ad,cd)+xol3(br,cr,dr,bc,bd,cd)-wol2(cr,dr,cd);

        return xol4(ar,br,cr,dr,ab,ac,ad,bc,bd,cd);
    }

DEVICE inline     OverlapReal xol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        bool abcf = flag4(ab,ac,ad,ar,bc,bd,br,cd,cr,dr);
        bool abdf = flag4(ab,ad,ac,ar,bd,bc,br,cd,dr,cr);
        bool acdf = flag4(ac,ad,ab,ar,cd,bc,cr,bd,dr,br);
        bool bcdf = flag4(bc,bd,ab,br,cd,ac,cr,ad,dr,ar);

        if(abcf && abdf && acdf && bcdf) return 0;

        if(abcf && abdf && acdf) return xol3(br,cr,dr,bc,bd,cd);
        if(abcf && abdf && bcdf) return xol3(ar,cr,dr,ac,ad,cd);
        if(abcf && acdf && bcdf) return xol3(ar,br,dr,ab,ad,bd);
        if(abdf && acdf && bcdf) return xol3(ar,br,cr,ab,ac,bc);

        if(abcf && abdf) return xol3(ar,cr,dr,ac,ad,cd)+xol3(br,cr,dr,bc,bd,cd)-wol2(cr,dr,cd);
        if(abcf && acdf) return xol3(ar,br,dr,ab,ad,bd)+xol3(br,cr,dr,bc,bd,cd)-wol2(br,dr,bd);
        if(abcf && bcdf) return xol3(ar,br,dr,ab,ad,bd)+xol3(ar,cr,dr,ac,ad,cd)-wol2(ar,dr,ad);
        if(abdf && acdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(br,cr,dr,bc,bd,cd)-wol2(br,cr,bc);
        if(abdf && bcdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,cr,dr,ac,ad,cd)-wol2(ar,cr,ac);
        if(acdf && bcdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,br,dr,ab,ad,bd)-wol2(ar,br,ab);

        if(abcf) return xol3(ar,br,dr,ab,ad,bd)+xol3(ar,cr,dr,ac,ad,cd)+xol3(br,cr,dr,bc,bd,cd)-wol2(ar,dr,ad)-wol2(br,dr,bd)-wol2(cr,dr,cd)+uol1(dr);
        if(abdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,cr,dr,ac,ad,cd)+xol3(br,cr,dr,bc,bd,cd)-wol2(ar,cr,ac)-wol2(br,cr,bc)-wol2(cr,dr,cd)+uol1(cr);
        if(acdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,br,dr,ab,ad,bd)+xol3(br,cr,dr,bc,bd,cd)-wol2(ar,br,ab)-wol2(br,cr,bc)-wol2(br,dr,bd)+uol1(br);
        if(bcdf) return xol3(ar,br,cr,ab,ac,bc)+xol3(ar,br,dr,ab,ad,bd)+xol3(ar,cr,dr,ac,ad,cd)-wol2(ar,br,ab)-wol2(ar,cr,ac)-wol2(ar,dr,ad)+uol1(ar);

        return yol4(ar,br,cr,dr,ab,ac,ad,bc,bd,cd);
    }

DEVICE inline     OverlapReal yol4(OverlapReal ar,OverlapReal br,OverlapReal cr,OverlapReal dr,OverlapReal ab,OverlapReal ac,OverlapReal ad,OverlapReal bc,OverlapReal bd,OverlapReal cd)
    {
        OverlapReal tad = acos(cos3(ad,ab,ac,bd,cd,bc)*sign(bc));
        OverlapReal tbd = acos(cos3(bd,ab,bc,ad,cd,ac)*sign(ac));
        OverlapReal tcd = acos(cos3(cd,ac,bc,ad,bd,ab)*sign(ab));
        OverlapReal tbc = acos(cos3(bc,ab,bd,ac,cd,ad)*sign(ad));
        OverlapReal tac = acos(cos3(ac,ab,ad,bc,cd,bd)*sign(bd));
        OverlapReal tab = acos(cos3(ab,ac,ad,bc,bd,cd)*sign(cd));

        return
        -t3(ab,ac,ad,bc,bd,cd)*sign(ar)*sign(br)*sign(cr)*sign(dr)
        +(xol3(ar,br,cr,ab,ac,bc)+
          xol3(ar,br,dr,ab,ad,bd)+
          xol3(ar,cr,dr,ac,ad,cd)+
          xol3(br,cr,dr,bc,bd,cd))/2
        -(wol2(ar,br,ab)*tab+
          wol2(ar,cr,ac)*tac+
          wol2(ar,dr,ad)*tad+
          wol2(br,cr,bc)*tbc+
          wol2(br,dr,bd)*tbd+
          wol2(cr,dr,cd)*tcd)/(2*M_PI)
        +(uol1(ar)*(tab+tac+tad-M_PI)+
          uol1(br)*(tab+tbc+tbd-M_PI)+
          uol1(cr)*(tac+tbc+tcd-M_PI)+
          uol1(dr)*(tad+tbd+tcd-M_PI))/(4*M_PI);
    }

} // detail
} // hpmc
#endif // __SPHINXOVERLAP__H__
