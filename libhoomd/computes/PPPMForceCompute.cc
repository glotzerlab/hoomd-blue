/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: sbarr

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
#include <boost/bind.hpp>

#include "PPPMForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>


using namespace boost;
using namespace boost::python;
using namespace std;


/*! \file PPPMForceCompute.cc
    \brief Contains code for the PPPMForceCompute class
*/

/*! \param sysdef System to compute forces on
    \param nlist Neighbor list
    \param group Particle group
    \post Memory is allocated, and forces are zeroed.
*/
PPPMForceCompute::PPPMForceCompute(boost::shared_ptr<SystemDefinition> sysdef, 
                                   boost::shared_ptr<NeighborList> nlist,
                                   boost::shared_ptr<ParticleGroup> group)
    : ForceCompute(sysdef), m_params_set(false), m_nlist(nlist), m_group(group),
      fft_in(NULL), fft_ex(NULL), fft_ey(NULL), fft_ez(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing PPPMForceCompute" << endl;

    assert(m_pdata);
    assert(m_nlist);

    m_box_changed = false;
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&PPPMForceCompute::slotBoxChanged, this));
    }

PPPMForceCompute::~PPPMForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PPPMForceCompute" << endl;

    if (fft_in)
        free(fft_in);
    if (fft_ex)
        free(fft_ex);
    if (fft_ey)
        free(fft_ey);
    if (fft_ez)
        free(fft_ez);

    m_boxchange_connection.disconnect();
    }

/*! \param Nx Number of grid points in x direction
    \param Ny Number of grid points in y direction
    \param Nz Number of grid points in z direction
    \param order Number of grid points in each direction to assign charges to
    \param kappa Screening parameter in erfc
    \param rcut Short-ranged cutoff, used for computing the relative force error

    Sets parameters for the long-ranged part of the electrostatics calculation
*/
void PPPMForceCompute::setParams(int Nx, int Ny, int Nz, int order, Scalar kappa, Scalar rcut)
    {
    m_params_set = true;
    m_Nx = Nx;
    m_Ny = Ny;
    m_Nz = Nz;
    m_order = order;
    m_kappa = kappa;
    m_rcut = rcut;
    first_run = 0;

    if(!(m_Nx == 2)&& !(m_Nx == 4)&& !(m_Nx == 8)&& !(m_Nx == 16)&& !(m_Nx == 32)&& !(m_Nx == 64)&& !(m_Nx == 128)&& !(m_Nx == 256)&& !(m_Nx == 512)&& !(m_Nx == 1024))
        {
        m_exec_conf->msg->warning() << "charge.pppm: PPPM X gridsize should be a power of 2 for the best performance" << endl;
        }
    if(!(m_Ny == 2)&& !(m_Ny == 4)&& !(m_Ny == 8)&& !(m_Ny == 16)&& !(m_Ny == 32)&& !(m_Ny == 64)&& !(m_Ny == 128)&& !(m_Ny == 256)&& !(m_Ny == 512)&& !(m_Ny == 1024))
        {
        m_exec_conf->msg->warning() << "charge.pppm: PPPM Y gridsize should be a power of 2 for the best performance" << endl;
        }
    if(!(m_Nz == 2)&& !(m_Nz == 4)&& !(m_Nz == 8)&& !(m_Nz == 16)&& !(m_Nz == 32)&& !(m_Nz == 64)&& !(m_Nz == 128)&& !(m_Nz == 256)&& !(m_Nz == 512)&& !(m_Nz == 1024))
        {
        m_exec_conf->msg->warning() << "charge.pppm: PPPM Z gridsize should be a power of 2 for the best performance" << endl;
        }
    if (m_order * (2*m_order +1) > CONSTANT_SIZE)
        {
        m_exec_conf->msg->error() << "charge.pppm: interpolation order too high, doesn't fit into constant array" << endl;
        throw std::runtime_error("Error initializing PPPMForceCompute");
        }
    if (m_order > MaxOrder)
        {
        m_exec_conf->msg->error() << "charge.pppm: interpolation order too high, max is " << MaxOrder << endl;
        throw std::runtime_error("Error initializing PPPMForceCompute");
        }

    GPUArray<cufftComplex> n_rho_real_space(Nx*Ny*Nz, exec_conf);
    m_rho_real_space.swap(n_rho_real_space);
    GPUArray<Scalar> n_green_hat(Nx*Ny*Nz, exec_conf);
    m_green_hat.swap(n_green_hat);

    GPUArray<Scalar> n_vg(6*Nx*Ny*Nz, exec_conf);
    m_vg.swap(n_vg);


    GPUArray<Scalar3> n_kvec(Nx*Ny*Nz, exec_conf);
    m_kvec.swap(n_kvec);
    GPUArray<cufftComplex> n_Ex(Nx*Ny*Nz, exec_conf);
    m_Ex.swap(n_Ex);
    GPUArray<cufftComplex> n_Ey(Nx*Ny*Nz, exec_conf);
    m_Ey.swap(n_Ey);
    GPUArray<cufftComplex> n_Ez(Nx*Ny*Nz, exec_conf);
    m_Ez.swap(n_Ez);
    GPUArray<Scalar> n_gf_b(order, exec_conf);
    m_gf_b.swap(n_gf_b);
    GPUArray<Scalar> n_rho_coeff(order*(2*order+1), exec_conf);
    m_rho_coeff.swap(n_rho_coeff);
    GPUArray<Scalar3> n_field(Nx*Ny*Nz, exec_conf);
    m_field.swap(n_field);
    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    // get system charge
    m_q = 0.f;
    m_q2 = 0.0;
    for(int i = 0; i < (int)m_pdata->getN(); i++) {
        m_q += h_charge.data[i];
        m_q2 += h_charge.data[i]*h_charge.data[i];
        }
    if(fabs(m_q) > 0.0)
        m_exec_conf->msg->warning() << "charge.pppm: system in not neutral, the net charge is " << m_q << endl;

    // compute RMS force error
    Scalar3 L = box.getL();
    Scalar hx =  L.x/(Scalar)Nx;
    Scalar hy =  L.y/(Scalar)Ny;
    Scalar hz =  L.z/(Scalar)Nz;
    Scalar lprx = PPPMForceCompute::rms(hx, L.x, (int)m_pdata->getN());
    Scalar lpry = PPPMForceCompute::rms(hy, L.y, (int)m_pdata->getN());
    Scalar lprz = PPPMForceCompute::rms(hz, L.z, (int)m_pdata->getN());
    Scalar lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
    Scalar spr = 2.0*m_q2*exp(-m_kappa*m_kappa*m_rcut*m_rcut) / sqrt((int)m_pdata->getN()*m_rcut*L.x*L.y*L.z);

    double RMS_error = MAX(lpr,spr);
    if(RMS_error > 0.1) {
        printf("!!!!!!!\n!!!!!!!\n!!!!!!!\nWARNING RMS error of %g is probably too high %f %f\n!!!!!!!\n!!!!!!!\n!!!!!!!\n", RMS_error, lpr, spr);
        }
    else{
        printf("Notice: PPPM RMS error: %g\n", RMS_error);
        }

     PPPMForceCompute::compute_rho_coeff();

    Scalar3 inverse_lattice_vector;
    Scalar invdet = 2.0f*M_PI/(L.x*L.y*L.z);
    inverse_lattice_vector.x = invdet*L.y*L.z;
    inverse_lattice_vector.y = invdet*L.x*L.z;
    inverse_lattice_vector.z = invdet*L.x*L.y;

    ArrayHandle<Scalar3> h_kvec(m_kvec, access_location::host, access_mode::readwrite);
    // Set up the k-vectors
    int ix, iy, iz, kper, lper, mper, k, l, m;
    for (ix = 0; ix < Nx; ix++) {
        Scalar3 j;
        j.x = ix > Nx/2 ? ix - Nx : ix;
        for (iy = 0; iy < Ny; iy++) {
            j.y = iy > Ny/2 ? iy - Ny : iy;
            for (iz = 0; iz < Nz; iz++) {
                j.z = iz > Nz/2 ? iz - Nz : iz;
                h_kvec.data[iz + Nz * (iy + Ny * ix)].x =  j.x*inverse_lattice_vector.x;
                h_kvec.data[iz + Nz * (iy + Ny * ix)].y =  j.y*inverse_lattice_vector.y;
                h_kvec.data[iz + Nz * (iy + Ny * ix)].z =  j.z*inverse_lattice_vector.z;
                }
            }
        }
 
    // Set up constants for virial calculation
    ArrayHandle<Scalar> h_vg(m_vg, access_location::host, access_mode::readwrite);;
    for(int x = 0; x < Nx; x++)
        {
        for(int y = 0; y < Ny; y++)
            {
            for(int z = 0; z < Nz; z++)
                {
                Scalar3 kvec = h_kvec.data[z + Nz * (y + Ny * x)];
                Scalar sqk =  kvec.x*kvec.x;
                sqk += kvec.y*kvec.y;
                sqk += kvec.z*kvec.z;
                int grid_point = z + Nz * (y + Ny * x);    
                if (sqk == 0.0) 
                    {
                    h_vg.data[0 + 6*grid_point] = 0.0f;
                    h_vg.data[1 + 6*grid_point] = 0.0f;
                    h_vg.data[2 + 6*grid_point] = 0.0f;
                    h_vg.data[3 + 6*grid_point] = 0.0f;
                    h_vg.data[4 + 6*grid_point] = 0.0f;
                    h_vg.data[5 + 6*grid_point] = 0.0f;
                   }
                else
                    {
                    Scalar vterm = -2.0 * (1.0/sqk + 0.25/(kappa*kappa));
                    h_vg.data[0 + 6*grid_point] =  1.0 + vterm*kvec.x*kvec.x;
                    h_vg.data[1 + 6*grid_point] =        vterm*kvec.x*kvec.y;
                    h_vg.data[2 + 6*grid_point] =        vterm*kvec.x*kvec.z;
                    h_vg.data[3 + 6*grid_point] =  1.0 + vterm*kvec.y*kvec.y;
                    h_vg.data[4 + 6*grid_point] =        vterm*kvec.y*kvec.z;
                    h_vg.data[5 + 6*grid_point] =  1.0 + vterm*kvec.z*kvec.z;
                    }
                } 
            } 
        }


    // Set up the grid based Green's function
    ArrayHandle<Scalar> h_green_hat(m_green_hat, access_location::host, access_mode::readwrite);
    Scalar snx, sny, snz, snx2, sny2, snz2;
    Scalar argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
    Scalar sum1, dot1, dot2;
    Scalar numerator, denominator, sqk;

    Scalar unitkx = (2.0*M_PI/L.x);
    Scalar unitky = (2.0*M_PI/L.y);
    Scalar unitkz = (2.0*M_PI/L.z);
   
    
    Scalar xprd = L.x; 
    Scalar yprd = L.y; 
    Scalar zprd_slab = L.z; 
    
    Scalar form = 1.0;

    PPPMForceCompute::compute_gf_denom();

    Scalar temp = floor(((kappa*xprd/(M_PI*Nx)) * 
                         pow(-log(EPS_HOC),0.25)));
    int nbx = (int)temp;

    temp = floor(((kappa*yprd/(M_PI*Ny)) * 
                  pow(-log(EPS_HOC),0.25)));
    int nby = (int)temp;

    temp =  floor(((kappa*zprd_slab/(M_PI*Nz)) * 
                   pow(-log(EPS_HOC),0.25)));
    int nbz = (int)temp;

    
    for (m = 0; m < Nz; m++) {
        mper = m - Nz*(2*m/Nz);
        snz = sin(0.5*unitkz*mper*zprd_slab/Nz);
        snz2 = snz*snz;

        for (l = 0; l < Ny; l++) {
            lper = l - Ny*(2*l/Ny);
            sny = sin(0.5*unitky*lper*yprd/Ny);
            sny2 = sny*sny;

            for (k = 0; k < Nx; k++) {
                kper = k - Nx*(2*k/Nx);
                snx = sin(0.5*unitkx*kper*xprd/Nx);
                snx2 = snx*snx;
      
                sqk = pow(Scalar(unitkx*kper),Scalar(2.0)) + pow(Scalar(unitky*lper),Scalar(2.0)) + 
                    pow(Scalar(unitkz*mper),Scalar(2.0));
                if (sqk != 0.0) {
                    numerator = form*12.5663706/sqk;
                    denominator = gf_denom(snx2,sny2,snz2);  

                    sum1 = 0.0;
                    for (ix = -nbx; ix <= nbx; ix++) {
                        qx = unitkx*(kper+(Scalar)(Nx*ix));
                        sx = exp(-.25*pow(Scalar(qx/kappa),Scalar(2.0)));
                        wx = 1.0;
                        argx = 0.5*qx*xprd/(Scalar)Nx;
                        if (argx != 0.0) wx = pow(sin(argx)/argx,order);
                        for (iy = -nby; iy <= nby; iy++) {
                            qy = unitky*(lper+(Scalar)(Ny*iy));
                            sy = exp(-.25*pow(Scalar(qy/kappa),Scalar(2.0)));
                            wy = 1.0;
                            argy = 0.5*qy*yprd/(Scalar)Ny;
                            if (argy != 0.0) wy = pow(sin(argy)/argy,order);
                            for (iz = -nbz; iz <= nbz; iz++) {
                                qz = unitkz*(mper+(Scalar)(Nz*iz));
                                sz = exp(-.25*pow(Scalar(qz/kappa),Scalar(2.0)));
                                wz = 1.0;
                                argz = 0.5*qz*zprd_slab/(Scalar)Nz;
                                if (argz != 0.0) wz = pow(sin(argz)/argz,order);

                                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                                dot2 = qx*qx+qy*qy+qz*qz;
                                sum1 += (dot1/dot2) * sx*sy*sz * pow(Scalar(wx*wy*wz),Scalar(2.0));
                                }
                            }
                        }
                    h_green_hat.data[m + Nz * (l + Ny * k)] = numerator*sum1/denominator;
                    } else h_green_hat.data[m + Nz * (l + Ny * k)] = 0.0;
                }
            }
        }
    Scalar scale = 1.0f/((Scalar)(Nx * Ny * Nz));
    m_energy_virial_factor = 0.5 * L.x * L.y * L.z * scale * scale;
    }

std::vector< std::string > PPPMForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pppm_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
  \param timestep Current time step of the simulation
*/
Scalar PPPMForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pppm_energy"))
        {
        compute(timestep);
        Scalar energy = calcEnergySum();
        return energy;
        }
    else
        {
        m_exec_conf->msg->error() << "charge.pppm: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
  \param timestep Current time step
*/

void PPPMForceCompute::computeForces(unsigned int timestep)
    {
    if (!m_params_set)
        {
        m_exec_conf->msg->error() << "charge.pppm: setParams must be called prior to computeForces()" << endl;
        throw std::runtime_error("Error computing forces in PPPMForceCompute");
        }
    
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("PPPM force");
    int dim[3];
    dim[0] = m_Nx;
    dim[1] = m_Ny;
    dim[2] = m_Nz;

    if(first_run == 0) 
        {
        first_run = 1;
        fft_in = (kiss_fft_cpx *)malloc(m_Nx*m_Ny*m_Nz*sizeof(kiss_fft_cpx));
        fft_ex = (kiss_fft_cpx *)malloc(m_Nx*m_Ny*m_Nz*sizeof(kiss_fft_cpx));
        fft_ey = (kiss_fft_cpx *)malloc(m_Nx*m_Ny*m_Nz*sizeof(kiss_fft_cpx));
        fft_ez = (kiss_fft_cpx *)malloc(m_Nx*m_Ny*m_Nz*sizeof(kiss_fft_cpx));

        fft_forward = kiss_fftnd_alloc(dim, 3, 0, NULL, NULL);
        fft_inverse = kiss_fftnd_alloc(dim, 3, 1, NULL, NULL);
        }

    if(m_box_changed)
        {
        const BoxDim& box = m_pdata->getBox();
        Scalar3 L = box.getL();
        PPPMForceCompute::reset_kvec_green_hat_cpu();
        Scalar scale = 1.0f/((Scalar)(m_Nx * m_Ny * m_Nz));
        m_energy_virial_factor = 0.5 * L.x * L.y * L.z * scale * scale;
        m_box_changed = false;
        }

    PPPMForceCompute::assign_charges_to_grid();

//FFTs go next
    
        { // scoping array handles
        ArrayHandle<cufftComplex> h_rho_real_space(m_rho_real_space, access_location::host, access_mode::readwrite);
        for(int i = 0; i < m_Nx * m_Ny * m_Nz ; i++) {
            fft_in[i].r = (float) h_rho_real_space.data[i].x;
            fft_in[i].i = (float)0.0;
            }

        kiss_fftnd(fft_forward, &fft_in[0], &fft_in[0]);

        for(int i = 0; i < m_Nx * m_Ny * m_Nz ; i++) {
            h_rho_real_space.data[i].x = fft_in[i].r;
            h_rho_real_space.data[i].y = fft_in[i].i;
    
            }
        }

    PPPMForceCompute::combined_green_e();

//More FFTs

        { // scoping array handles
        ArrayHandle<cufftComplex> h_Ex(m_Ex, access_location::host, access_mode::readwrite);
        ArrayHandle<cufftComplex> h_Ey(m_Ey, access_location::host, access_mode::readwrite);
        ArrayHandle<cufftComplex> h_Ez(m_Ez, access_location::host, access_mode::readwrite);

        for(int i = 0; i < m_Nx * m_Ny * m_Nz ; i++)
            {
            fft_ex[i].r = (float) h_Ex.data[i].x;
            fft_ex[i].i = (float) h_Ex.data[i].y;

            fft_ey[i].r = (float) h_Ey.data[i].x;
            fft_ey[i].i = (float) h_Ey.data[i].y;

            fft_ez[i].r = (float) h_Ez.data[i].x;
            fft_ez[i].i = (float) h_Ez.data[i].y;
            }


        kiss_fftnd(fft_inverse, &fft_ex[0], &fft_ex[0]);
        kiss_fftnd(fft_inverse, &fft_ey[0], &fft_ey[0]);
        kiss_fftnd(fft_inverse, &fft_ez[0], &fft_ez[0]);

        for(int i = 0; i < m_Nx * m_Ny * m_Nz ; i++)
            {
            h_Ex.data[i].x = fft_ex[i].r;
            h_Ex.data[i].y = fft_ex[i].i;

            h_Ey.data[i].x = fft_ey[i].r;
            h_Ey.data[i].y = fft_ey[i].i;

            h_Ez.data[i].x = fft_ez[i].r;
            h_Ez.data[i].y = fft_ez[i].i;
            }
        }

    PPPMForceCompute::calculate_forces();

    // If there are exclusions, correct for the long-range part of the potential
    if( m_nlist->getExclusionsSet()) 
        {
        PPPMForceCompute::fix_exclusions_cpu();
        }

    // access flags and correct energy and virial if needed
    PDataFlags flags = this->m_pdata->getFlags();
    if (flags[pdata_flag::potential_energy] || flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial])
        {
        fix_thermo_quantities();
        }

    if (m_prof) m_prof->pop();
    }

Scalar PPPMForceCompute::rms(Scalar h, Scalar prd, Scalar natoms)
    {
    int m;
    Scalar sum = 0.0;
    Scalar acons[8][7]; 

    acons[1][0] = 2.0 / 3.0;
    acons[2][0] = 1.0 / 50.0;
    acons[2][1] = 5.0 / 294.0;
    acons[3][0] = 1.0 / 588.0;
    acons[3][1] = 7.0 / 1440.0;
    acons[3][2] = 21.0 / 3872.0;
    acons[4][0] = 1.0 / 4320.0;
    acons[4][1] = 3.0 / 1936.0;
    acons[4][2] = 7601.0 / 2271360.0;
    acons[4][3] = 143.0 / 28800.0;
    acons[5][0] = 1.0 / 23232.0;
    acons[5][1] = 7601.0 / 13628160.0;
    acons[5][2] = 143.0 / 69120.0;
    acons[5][3] = 517231.0 / 106536960.0;
    acons[5][4] = 106640677.0 / 11737571328.0;
    acons[6][0] = 691.0 / 68140800.0;
    acons[6][1] = 13.0 / 57600.0;
    acons[6][2] = 47021.0 / 35512320.0;
    acons[6][3] = 9694607.0 / 2095994880.0;
    acons[6][4] = 733191589.0 / 59609088000.0;
    acons[6][5] = 326190917.0 / 11700633600.0;
    acons[7][0] = 1.0 / 345600.0;
    acons[7][1] = 3617.0 / 35512320.0;
    acons[7][2] = 745739.0 / 838397952.0;
    acons[7][3] = 56399353.0 / 12773376000.0;
    acons[7][4] = 25091609.0 / 1560084480.0;
    acons[7][5] = 1755948832039.0 / 36229939200000.0;
    acons[7][6] = 4887769399.0 / 37838389248.0;

    for (m = 0; m < m_order; m++) 
        sum += acons[m_order][m] * pow(h*m_kappa,2.0f*(Scalar)m);
    Scalar value = m_q2 * pow(h*m_kappa,(Scalar)m_order) *
        sqrt(m_kappa*prd*sqrt(2.0*M_PI)*sum/natoms) / (prd*prd);
    return value;
    }


void PPPMForceCompute::compute_rho_coeff()
    {
    int j, k, l, m;
    Scalar s;
    Scalar a[136]; 
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::readwrite);

    //    usage: a[x][y] = a[y + x*(2*m_order+1)]
    
    for(l=0; l<m_order; l++)
        {
        for(m=0; m<(2*m_order+1); m++)
            {
            a[m + l*(2*m_order +1)] = 0.0f;
            }
        }

    for (k = -m_order; k <= m_order; k++) 
        for (l = 0; l < m_order; l++) {
            a[(k+m_order) + l * (2*m_order+1)] = 0.0f;
            }

    a[m_order + 0 * (2*m_order+1)] = 1.0f;
    for (j = 1; j < m_order; j++) {
        for (k = -j; k <= j; k += 2) {
            s = 0.0;
            for (l = 0; l < j; l++) {
                a[(k + m_order) + (l+1)*(2*m_order+1)] = (a[(k+1+m_order) + l * (2*m_order + 1)] - a[(k-1+m_order) + l * (2*m_order + 1)]) / (l+1);
                s += pow(0.5,(double) (l+1)) * (a[(k-1+m_order) + l * (2*m_order + 1)] + pow(-1.0,(double) l) * a[(k+1+m_order) + l * (2*m_order + 1)] ) / (double)(l+1);
                }
            a[k+m_order + 0 * (2*m_order+1)] = s;
            }
        }

    m = 0;
    for (k = -(m_order-1); k < m_order; k += 2) {
        for (l = 0; l < m_order; l++) {
            h_rho_coeff.data[m + l*(2*m_order +1)] = a[k+m_order + l * (2*m_order + 1)];
            }
        m++;
        }
    }

void PPPMForceCompute::compute_gf_denom()
    {
    int k,l,m;
      ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::readwrite);
    for (l = 1; l < m_order; l++) h_gf_b.data[l] = 0.0;
    h_gf_b.data[0] = 1.0;
  
    for (m = 1; m < m_order; m++) {
        for (l = m; l > 0; l--) {
            h_gf_b.data[l] = 4.0 * (h_gf_b.data[l]*(l-m)*(l-m-0.5)-h_gf_b.data[l-1]*(l-m-1)*(l-m-1));
            }
        h_gf_b.data[0] = 4.0 * (h_gf_b.data[0]*(l-m)*(l-m-0.5));
    }

    int ifact = 1;
    for (k = 1; k < 2*m_order; k++) ifact *= k;
    Scalar gaminv = 1.0/ifact;
    for (l = 0; l < m_order; l++) h_gf_b.data[l] *= gaminv;
    }

Scalar PPPMForceCompute::gf_denom(Scalar x, Scalar y, Scalar z)
    {
    int l ;
    Scalar sx,sy,sz;
    ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::readwrite);
    sz = sy = sx = 0.0;
    for (l = m_order-1; l >= 0; l--) {
        sx = h_gf_b.data[l] + sx*x;
        sy = h_gf_b.data[l] + sy*y;
        sz = h_gf_b.data[l] + sz*z;
        }
    Scalar s = sx*sy*sz;
    return s*s;
    }


void PPPMForceCompute::reset_kvec_green_hat_cpu()
    {
    ArrayHandle<Scalar3> h_kvec(m_kvec, access_location::host, access_mode::readwrite);
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();

    Scalar3 inverse_lattice_vector;
    Scalar invdet = 2.0f*M_PI/(L.x*L.y*L.z);
    inverse_lattice_vector.x = invdet*L.y*L.z;
    inverse_lattice_vector.y = invdet*L.x*L.z;
    inverse_lattice_vector.z = invdet*L.x*L.y;

    // Set up the k-vectors
    int ix, iy, iz, kper, lper, mper, k, l, m;
    for (ix = 0; ix < m_Nx; ix++) {
        Scalar3 j;
        j.x = ix > m_Nx/2 ? ix - m_Nx : ix;
        for (iy = 0; iy < m_Ny; iy++) {
            j.y = iy > m_Ny/2 ? iy - m_Ny : iy;
            for (iz = 0; iz < m_Nz; iz++) {
                j.z = iz > m_Nz/2 ? iz - m_Nz : iz;
                h_kvec.data[iz + m_Nz * (iy + m_Ny * ix)].x =  j.x*inverse_lattice_vector.x;
                h_kvec.data[iz + m_Nz * (iy + m_Ny * ix)].y =  j.y*inverse_lattice_vector.y;
                h_kvec.data[iz + m_Nz * (iy + m_Ny * ix)].z =  j.z*inverse_lattice_vector.z;
                }
            }
        }
 
    // Set up constants for virial calculation
    ArrayHandle<Scalar> h_vg(m_vg, access_location::host, access_mode::readwrite);;
    for(int x = 0; x < m_Nx; x++)
        {
        for(int y = 0; y < m_Ny; y++)
            {
            for(int z = 0; z < m_Nz; z++)
                {
                Scalar3 kvec = h_kvec.data[z + m_Nz * (y + m_Ny * x)];
                Scalar sqk =  kvec.x*kvec.x;
                sqk += kvec.y*kvec.y;
                sqk += kvec.z*kvec.z;
    
                int grid_point = z + m_Nz * (y + m_Ny * x);    
                if (sqk == 0.0) 
                    {
                    h_vg.data[0 + 6*grid_point] = 0.0f;
                    h_vg.data[1 + 6*grid_point] = 0.0f;
                    h_vg.data[2 + 6*grid_point] = 0.0f;
                    h_vg.data[3 + 6*grid_point] = 0.0f;
                    h_vg.data[4 + 6*grid_point] = 0.0f;
                    h_vg.data[5 + 6*grid_point] = 0.0f;
                    }
                else
                    {
                    Scalar vterm = -2.0 * (1.0/sqk + 0.25/(m_kappa*m_kappa));
                    h_vg.data[0 + 6*grid_point] =  1.0 + vterm*kvec.x*kvec.x;
                    h_vg.data[1 + 6*grid_point] =        vterm*kvec.x*kvec.y;
                    h_vg.data[2 + 6*grid_point] =        vterm*kvec.x*kvec.z;
                    h_vg.data[3 + 6*grid_point] =  1.0 + vterm*kvec.y*kvec.y;
                    h_vg.data[4 + 6*grid_point] =        vterm*kvec.y*kvec.z;
                    h_vg.data[5 + 6*grid_point] =  1.0 + vterm*kvec.z*kvec.z;
                    }
                } 
            } 
        }


    // Set up the grid based Green's function
    ArrayHandle<Scalar> h_green_hat(m_green_hat, access_location::host, access_mode::readwrite);
    Scalar snx, sny, snz, snx2, sny2, snz2;
    Scalar argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
    Scalar sum1, dot1, dot2;
    Scalar numerator, denominator, sqk;

    Scalar unitkx = (2.0*M_PI/L.x);
    Scalar unitky = (2.0*M_PI/L.y);
    Scalar unitkz = (2.0*M_PI/L.z);
   
    
    Scalar xprd = L.x; 
    Scalar yprd = L.y; 
    Scalar zprd_slab = L.z; 
    
    Scalar form = 1.0;

    PPPMForceCompute::compute_gf_denom();

    Scalar temp = floor(((m_kappa*xprd/(M_PI*m_Nx)) * 
                         pow(-log(EPS_HOC),0.25)));
    int nbx = (int)temp;

    temp = floor(((m_kappa*yprd/(M_PI*m_Ny)) * 
                  pow(-log(EPS_HOC),0.25)));
    int nby = (int)temp;

    temp =  floor(((m_kappa*zprd_slab/(M_PI*m_Nz)) * 
                   pow(-log(EPS_HOC),0.25)));
    int nbz = (int)temp;

    
    for (m = 0; m < m_Nz; m++) {
        mper = m - m_Nz*(2*m/m_Nz);
        snz = sin(0.5*unitkz*mper*zprd_slab/m_Nz);
        snz2 = snz*snz;

        for (l = 0; l < m_Ny; l++) {
            lper = l - m_Ny*(2*l/m_Ny);
            sny = sin(0.5*unitky*lper*yprd/m_Ny);
            sny2 = sny*sny;

            for (k = 0; k < m_Nx; k++) {
                kper = k - m_Nx*(2*k/m_Nx);
                snx = sin(0.5*unitkx*kper*xprd/m_Nx);
                snx2 = snx*snx;
      
                sqk = pow(Scalar(unitkx*kper),Scalar(2.0)) + pow(Scalar(unitky*lper),Scalar(2.0)) + 
                    pow(Scalar(unitkz*mper),Scalar(2.0));
                if (sqk != 0.0) {
                    numerator = form*12.5663706/sqk;
                    denominator = gf_denom(snx2,sny2,snz2);  

                    sum1 = 0.0;
                    for (ix = -nbx; ix <= nbx; ix++) {
                        qx = unitkx*(kper+(Scalar)(m_Nx*ix));
                        sx = exp(-.25*pow(Scalar(qx/m_kappa),Scalar(2.0)));
                        wx = 1.0;
                        argx = 0.5*qx*xprd/(Scalar)m_Nx;
                        if (argx != 0.0) wx = pow(sin(argx)/argx,m_order);
                        for (iy = -nby; iy <= nby; iy++) {
                            qy = unitky*(lper+(Scalar)(m_Ny*iy));
                            sy = exp(-.25*pow(Scalar(qy/m_kappa),Scalar(2.0)));
                            wy = 1.0;
                            argy = 0.5*qy*yprd/(Scalar)m_Ny;
                            if (argy != 0.0) wy = pow(sin(argy)/argy,m_order);
                            for (iz = -nbz; iz <= nbz; iz++) {
                                qz = unitkz*(mper+(Scalar)(m_Nz*iz));
                                sz = exp(-.25*pow(Scalar(qz/m_kappa),Scalar(2.0)));
                                wz = 1.0;
                                argz = 0.5*qz*zprd_slab/(Scalar)m_Nz;
                                if (argz != 0.0) wz = pow(sin(argz)/argz,m_order);

                                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                                dot2 = qx*qx+qy*qy+qz*qz;
                                sum1 += (dot1/dot2) * sx*sy*sz * pow(Scalar(wx*wy*wz),Scalar(2.0));
                                }
                            }
                        }
                    h_green_hat.data[m + m_Nz * (l + m_Ny * k)] = numerator*sum1/denominator;
                    } else h_green_hat.data[m + m_Nz * (l + m_Ny * k)] = 0.0;
                }
            }
        }
    }    

void PPPMForceCompute::assign_charges_to_grid()
    {

    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::read);
    ArrayHandle<cufftComplex> h_rho_real_space(m_rho_real_space, access_location::host, access_mode::readwrite);

    memset(h_rho_real_space.data, 0, sizeof(cufftComplex)*m_Nx*m_Ny*m_Nz);

    for(int i = 0; i < (int)m_pdata->getN(); i++)
        {
        Scalar qi = h_charge.data[i];
        Scalar4 posi;
        posi.x = h_pos.data[i].x;
        posi.y = h_pos.data[i].y;
        posi.z = h_pos.data[i].z;

        Scalar box_dx = L.x / ((Scalar)m_Nx);
        Scalar box_dy = L.y / ((Scalar)m_Ny);
        Scalar box_dz = L.z / ((Scalar)m_Nz);
 
        //normalize position to gridsize:
        posi.x += L.x / 2.0f;
        posi.y += L.y / 2.0f;
        posi.z += L.z / 2.0f;
   
        posi.x /= box_dx;
        posi.y /= box_dy;
        posi.z /= box_dz;


        Scalar shift, shiftone, x0, y0, z0, dx, dy, dz;
        int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
        nlower = -(m_order-1)/2;
        nupper = m_order/2;
    
        if (m_order % 2) 
            {
            shift =0.5;
            shiftone = 0.0;
            }
        else 
            {
            shift = 0.0;
            shiftone = 0.5;
            }

        nxi = (int)(posi.x + shift);
        nyi = (int)(posi.y + shift);
        nzi = (int)(posi.z + shift);
 
        dx = shiftone+(Scalar)nxi-posi.x;
        dy = shiftone+(Scalar)nyi-posi.y;
        dz = shiftone+(Scalar)nzi-posi.z;

        int n,m,l,k;
        Scalar result;
        int mult_fact = 2*m_order+1;

        x0 = qi / (box_dx*box_dy*box_dz);
        for (n = nlower; n <= nupper; n++) {
            mx = n+nxi;
            if(mx >= m_Nx) mx -= m_Nx;
            if(mx < 0)  mx += m_Nx;
            result = 0.0f;
            for (k = m_order-1; k >= 0; k--) {
                result = h_rho_coeff.data[n-nlower + k*mult_fact] + result * dx;
                }
            y0 = x0*result;
            for (m = nlower; m <= nupper; m++) {
                my = m+nyi;
                if(my >= m_Ny) my -= m_Ny;
                if(my < 0)  my += m_Ny;
                result = 0.0f;
                for (k = m_order-1; k >= 0; k--) {
                    result = h_rho_coeff.data[m-nlower + k*mult_fact] + result * dy;
                    }
                z0 = y0*result;
                for (l = nlower; l <= nupper; l++) {
                    mz = l+nzi;
                    if(mz >= m_Nz) mz -= m_Nz;
                    if(mz < 0)  mz += m_Nz;
                    result = 0.0f;
                    for (k = m_order-1; k >= 0; k--) {
                        result = h_rho_coeff.data[l-nlower + k*mult_fact] + result * dz;
                        }
                    h_rho_real_space.data[mz + m_Nz * (my + m_Ny * mx)].x += z0*result;
                    }
                }
            }
        }
    
    }

void PPPMForceCompute::combined_green_e()
    {

    ArrayHandle<Scalar3> h_kvec(m_kvec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_green_hat(m_green_hat, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_Ex(m_Ex, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_Ey(m_Ey, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_Ez(m_Ez, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_rho_real_space(m_rho_real_space, access_location::host, access_mode::readwrite);

    unsigned int NNN = m_Nx*m_Ny*m_Nz;
    for(unsigned int i = 0; i < NNN; i++)
        {

        cufftComplex rho_local = h_rho_real_space.data[i];
        Scalar scale_times_green = h_green_hat.data[i] / ((Scalar)(NNN));
        rho_local.x *= scale_times_green;
        rho_local.y *= scale_times_green;

        h_Ex.data[i].x = h_kvec.data[i].x * rho_local.y;
        h_Ex.data[i].y = -h_kvec.data[i].x * rho_local.x;
    
        h_Ey.data[i].x = h_kvec.data[i].y * rho_local.y;
        h_Ey.data[i].y = -h_kvec.data[i].y * rho_local.x;
    
        h_Ez.data[i].x = h_kvec.data[i].z * rho_local.y;
        h_Ez.data[i].y = -h_kvec.data[i].z * rho_local.x;
        }
    }

void PPPMForceCompute::calculate_forces()
    {
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();
    
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::read);
    ArrayHandle<cufftComplex> h_Ex(m_Ex, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_Ey(m_Ey, access_location::host, access_mode::readwrite);
    ArrayHandle<cufftComplex> h_Ez(m_Ez, access_location::host, access_mode::readwrite);

    for(int i = 0; i < (int)m_pdata->getN(); i++)
        {
        Scalar qi = h_charge.data[i];
        Scalar4 posi;
        posi.x = h_pos.data[i].x;
        posi.y = h_pos.data[i].y;
        posi.z = h_pos.data[i].z;

        Scalar box_dx = L.x / ((Scalar)m_Nx);
        Scalar box_dy = L.y / ((Scalar)m_Ny);
        Scalar box_dz = L.z / ((Scalar)m_Nz);
 
        //normalize position to gridsize:
        posi.x += L.x / 2.0f;
        posi.y += L.y / 2.0f;
        posi.z += L.z / 2.0f;
   
        posi.x /= box_dx;
        posi.y /= box_dy;
        posi.z /= box_dz;


        Scalar shift, shiftone, x0, y0, z0, dx, dy, dz;
        int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
        nlower = -(m_order-1)/2;
        nupper = m_order/2;
    
        if (m_order % 2) 
            {
            shift =0.5;
            shiftone = 0.0;
            }
        else 
            {
            shift = 0.0;
            shiftone = 0.5;
            }

        nxi = (int)(posi.x + shift);
        nyi = (int)(posi.y + shift);
        nzi = (int)(posi.z + shift);
 
        dx = shiftone+(Scalar)nxi-posi.x;
        dy = shiftone+(Scalar)nyi-posi.y;
        dz = shiftone+(Scalar)nzi-posi.z;

        int n,m,l,k;
        Scalar result;
        int mult_fact = 2*m_order+1;
        for (n = nlower; n <= nupper; n++) {
            mx = n+nxi;
            if(mx >= m_Nx) mx -= m_Nx;
            if(mx < 0)  mx += m_Nx;
            result = 0.0f;
            for (k = m_order-1; k >= 0; k--) {
                result = h_rho_coeff.data[n-nlower + k*mult_fact] + result * dx;
                }
            x0 = result;
            for (m = nlower; m <= nupper; m++) {
                my = m+nyi;
                if(my >= m_Ny) my -= m_Ny;
                if(my < 0)  my += m_Ny;
                result = 0.0f;
                for (k = m_order-1; k >= 0; k--) {
                    result = h_rho_coeff.data[m-nlower + k*mult_fact] + result * dy;
                    }
                y0 = x0*result;
                for (l = nlower; l <= nupper; l++) {
                    mz = l+nzi;
                    if(mz >= m_Nz) mz -= m_Nz;
                    if(mz < 0)  mz += m_Nz;
                    result = 0.0f;
                    for (k = m_order-1; k >= 0; k--) {
                        result = h_rho_coeff.data[l-nlower + k*mult_fact] + result * dz;
                        }
                    z0 = y0*result;
                    Scalar local_field_x = h_Ex.data[mz + m_Nz * (my + m_Ny * mx)].x;
                    Scalar local_field_y = h_Ey.data[mz + m_Nz * (my + m_Ny * mx)].x;
                    Scalar local_field_z = h_Ez.data[mz + m_Nz * (my + m_Ny * mx)].x;
                    h_force.data[i].x += qi*z0*local_field_x;
                    h_force.data[i].y += qi*z0*local_field_y;
                    h_force.data[i].z += qi*z0*local_field_z;
                    }
                }
            }
        }
    
    }

void PPPMForceCompute::fix_exclusions_cpu()
    {
    unsigned int group_size = m_group->getNumMembers();
    // just drop out if the group is an empty group
    if (group_size == 0)
        return;

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::readwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);

    ArrayHandle< unsigned int > d_group_members(m_group->getIndexArray(), access_location::host, access_mode::read);
    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<unsigned int> d_exlist(m_nlist->getExListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> d_n_ex(m_nlist->getNExArray(), access_location::host, access_mode::read);
    Index2D nex = m_nlist->getExListIndexer();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    for(unsigned int i = 0; i < group_size; i++)
        {
        Scalar4 force = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
        Scalar virial[6];
        for (unsigned int k = 0; k < 6; k++)
            virial[k] = 0.0f;
        unsigned int idx = d_group_members.data[i];
        Scalar3 posi;
        posi.x = h_pos.data[idx].x;
        posi.y = h_pos.data[idx].y;
        posi.z = h_pos.data[idx].z;
        Scalar qi = h_charge.data[idx];

        unsigned int n_neigh = d_n_ex.data[idx];
        const Scalar sqrtpi = sqrtf(M_PI);
        unsigned int cur_j = 0;

        for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
            {
            cur_j = d_exlist.data[nex(idx, neigh_idx)];
           // get the neighbor's position
            Scalar3 posj;
            posj.x = h_pos.data[cur_j].x;
            posj.y = h_pos.data[cur_j].y;
            posj.z = h_pos.data[cur_j].z;
            Scalar qj = h_charge.data[cur_j];
            Scalar3 dx = posi - posj;
            
            // apply periodic boundary conditions: 
            dx = box.minImage(dx);
            
            Scalar rsq = dot(dx, dx);
            Scalar r = sqrtf(rsq);
            Scalar qiqj = qi * qj;
            Scalar erffac = erf(m_kappa * r) / r;
            Scalar force_divr = qiqj * (-2.0f * exp(-rsq * m_kappa * m_kappa) * m_kappa / (sqrtpi * rsq) + erffac / rsq);
            Scalar pair_eng = qiqj * erffac; 
            virial[0]+= Scalar(0.5) * dx.x * dx.x * force_divr;
            virial[1]+= Scalar(0.5) * dx.y * dx.x * force_divr;
            virial[2]+= Scalar(0.5) * dx.z * dx.x * force_divr;
            virial[3]+= Scalar(0.5) * dx.y * dx.y * force_divr;
            virial[4]+= Scalar(0.5) * dx.z * dx.y * force_divr;
            virial[5]+= Scalar(0.5) * dx.z * dx.z * force_divr;
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;
            force.w += pair_eng;
            }
        force.w *= 0.5f;
        h_force.data[idx].x -= force.x;
        h_force.data[idx].y -= force.y;
        h_force.data[idx].z -= force.z;
        h_force.data[idx].w = -force.w;
        for (unsigned int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+idx] = -virial[k];
        }
    
    }

/*! Computes the additional energy and virial contributed by PPPM
    \note The additional terms are simply added onto particle 0 so that they will be accounted for by
    ComputeThermo
*/
void PPPMForceCompute::fix_thermo_quantities()
    {
    // access data arrays
    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();

    ArrayHandle<cufftComplex> d_rho_real_space(m_rho_real_space, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> d_green_hat(m_green_hat, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> d_vg(m_vg, access_location::host, access_mode::readwrite);
    Scalar2 pppm_virial_energy = make_scalar2(0.0, 0.0);

    Scalar v_xx=0.0, v_xy=0.0, v_xz=0.0, v_yy=0.0, v_yz=0.0, v_zz=0.0;



    // compute the correction
    for (int i = 0; i < m_Nx*m_Ny*m_Nz; i++)
        {
        Scalar energy = d_green_hat.data[i]*(d_rho_real_space.data[i].x*d_rho_real_space.data[i].x +
                                             d_rho_real_space.data[i].y*d_rho_real_space.data[i].y);
        Scalar pressure = energy*(d_vg.data[0+6*i] + d_vg.data[3+6*i] + d_vg.data[5+6*i]);
        v_xx += d_vg.data[0+6*i]*energy;
        v_xy += d_vg.data[1+6*i]*energy;
        v_xz += d_vg.data[2+6*i]*energy;
        v_yy += d_vg.data[3+6*i]*energy;
        v_yz += d_vg.data[4+6*i]*energy;
        v_zz += d_vg.data[5+6*i]*energy;
        pppm_virial_energy.x += pressure;
        pppm_virial_energy.y += energy;
        }

    pppm_virial_energy.x *= m_energy_virial_factor/ (3.0f * L.x * L.y * L.z);
    pppm_virial_energy.y *= m_energy_virial_factor;
    pppm_virial_energy.y -= m_q2 * m_kappa / 1.772453850905516027298168f;
    pppm_virial_energy.y -= 0.5*M_PI*m_q*m_q / (m_kappa*m_kappa* L.x * L.y * L.z);

    // apply the correction to particle 0
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::readwrite);
    h_force.data[0].w += pppm_virial_energy.y;
    

    // Compute full virial tensor
    unsigned int virial_pitch = m_virial.getPitch();
    h_virial.data[0*virial_pitch+0] += v_xx*m_energy_virial_factor;
    h_virial.data[1*virial_pitch+0] += v_xy*m_energy_virial_factor;
    h_virial.data[2*virial_pitch+0] += v_xz*m_energy_virial_factor;
    h_virial.data[3*virial_pitch+0] += v_yy*m_energy_virial_factor;
    h_virial.data[4*virial_pitch+0] += v_yz*m_energy_virial_factor;
    h_virial.data[5*virial_pitch+0] += v_zz*m_energy_virial_factor;
    }

void export_PPPMForceCompute()
    {
    class_<PPPMForceCompute, boost::shared_ptr<PPPMForceCompute>, bases<ForceCompute>, boost::noncopyable >
        ("PPPMForceCompute", init< boost::shared_ptr<SystemDefinition>, 
         boost::shared_ptr<NeighborList>,
         boost::shared_ptr<ParticleGroup> >())
        .def("setParams", &PPPMForceCompute::setParams)
        ;
    }
