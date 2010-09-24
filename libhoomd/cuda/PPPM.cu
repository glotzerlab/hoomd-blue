#include "PPPM.cuh"

extern struct electrostatics_data es_data;

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MaxOrder 12
#define CONSTANT_SIZE 2048

//! Texture for reading particle positions	
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;		

//! Texture for reading particle charges
texture<float, 1, cudaReadModeElementType> pdata_charge_tex;

//! Constant memory for gridpoint weighting
__device__ __constant__ float weight_factors[CONSTANT_SIZE];


typedef struct { float xx, yx, zx, xy, yy, zy, xz, yz, zz; } tensor_t;

struct coord
{
  float x;
  float y;
  float z;
};

typedef struct { float x, y, z; } cart_t;


double rms(double h, double prd, double natoms, int order, double kappa, double q2)
{
    int m;
    double sum = 0.0;
    double acons[8][7]; 

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

    for (m = 0; m < order; m++) 
	sum += acons[order][m] * pow(h*kappa,2.0*m);
    double value = q2 * pow(h*kappa,order) *
	sqrt(kappa*prd*sqrt(2.0*M_PI)*sum/natoms) / (prd*prd);
    return value;
}

void scalar_multiply(tensor_t *a, float b)
{
  a->xx *= b;  a->xy *= b;  a->xz *= b;
  a->yx *= b;  a->yy *= b;  a->yz *= b;
  a->zx *= b;  a->zy *= b;  a->zz *= b;
}
  
  
void transpose(tensor_t *a, tensor_t *b)
{
  a->xx = b->xx;  a->xy = b->yx;  a->xz = b->zx;
  a->yx = b->xy;  a->yy = b->yy;  a->yz = b->zy;
  a->zx = b->xz;  a->zy = b->yz;  a->zz = b->zz;
}

float det(tensor_t *a)
{
  return -a->xz*a->yy*a->zx + a->xy*a->yz*a->zx + 
    a->xz*a->yx*a->zy - a->xx*a->yz*a->zy - 
    a->xy*a->yx*a->zz + a->xx*a->yy*a->zz;
}

void inverse(tensor_t *a, tensor_t *b)
{
  float invdet = 1.0/det(b);
  a->xx = invdet*(-b->yz*b->zy + b->yy*b->zz);
  a->xy = invdet*( b->xz*b->zy - b->xy*b->zz);
  a->xz = invdet*(-b->xz*b->yy + b->xy*b->yz);
  a->yx = invdet*( b->yz*b->zx - b->yx*b->zz);
  a->yy = invdet*(-b->xz*b->zx + b->xx*b->zz);
  a->yz = invdet*( b->xz*b->yx - b->xx*b->yz);
  a->zx = invdet*(-b->yy*b->zx + b->yx*b->zy);
  a->zy = invdet*( b->xy*b->zx - b->xx*b->zy);
  a->zz = invdet*(-b->xy*b->yx + b->xx*b->yy);
}

void multiply(float3 *a, tensor_t *t, struct coord *c)
{
  a->x = t->xx*c->x + t->xy*c->y + t->xz*c->z;
  a->y = t->yx*c->x + t->yy*c->y + t->yz*c->z;
  a->z = t->zx*c->x + t->zy*c->y + t->zz*c->z;
}

static float *gf_b;

float gf_denom(float x, float y, float z, int order)
{
  int l ;
  float sx,sy,sz;
  sz = sy = sx = 0.0;
  for (l = order-1; l >= 0; l--) {
    sx = gf_b[l] + sx*x;
    sy = gf_b[l] + sy*y;
    sz = gf_b[l] + sz*z;
  }
  float s = sx*sy*sz;
  return s*s;
}

void compute_gf_denom(int order)
{
  int k,l,m;
  
  for (l = 1; l < order; l++) gf_b[l] = 0.0;
  gf_b[0] = 1.0;
  
  for (m = 1; m < order; m++) {
    for (l = m; l > 0; l--) 
      gf_b[l] = 4.0 * (gf_b[l]*(l-m)*(l-m-0.5)-gf_b[l-1]*(l-m-1)*(l-m-1));
    gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5));
  }

  int ifact = 1;
  for (k = 1; k < 2*order; k++) ifact *= k;
  float gaminv = 1.0/ifact;
  for (l = 0; l < order; l++) gf_b[l] *= gaminv;
}


void compute_rho_coeff(int assignment_order, float* rho_coeff)
{
  int j, k, l, m;
  float s;
  int order = assignment_order;
  float *a = (float*)malloc(order * (2*order+1) * sizeof(float)); 
  //    usage: a[x][y] = a[y + x*(2*order+1)]
    
  for(l=0; l<order; l++)
    {
      for(m=0; m<(2*order+1); m++)
	{
	  rho_coeff[m + l*(2*order +1)] = 0.0f;
	}
    }

  for (k = -order; k <= order; k++) 
    for (l = 0; l < order; l++) {
      a[(k+order) + l * (2*order+1)] = 0.0f;
    }

  a[order + 0 * (2*order+1)] = 1.0f;
  for (j = 1; j < order; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
	a[(k + order) + (l+1)*(2*order+1)] = (a[(k+1+order) + l * (2*order + 1)] - a[(k-1+order) + l * (2*order + 1)]) / (l+1);
	s += pow(0.5,(double) (l+1)) * (a[(k-1+order) + l * (2*order + 1)] + pow(-1.0,(double) l) * a[(k+1+order) + l * (2*order + 1)] ) / (double)(l+1);
      }
      a[k+order + 0 * (2*order+1)] = s;
    }
  }

  m = 0;
  for (k = -(order-1); k < order; k += 2) {
    for (l = 0; l < order; l++) {
      rho_coeff[m + l*(2*order +1)] = a[k+order + l * (2*order + 1)];
    }
    m++;
  }
  free(a);
}


__global__ void copy_data_kernel(gpu_boxsize box_old, gpu_boxsize *box_new)
{
  box_new[0].Lx = box_old.Lx;
  box_new[0].Ly = box_old.Ly;
  box_new[0].Lz = box_old.Lz;
}

void electrostatics_allocation(const gpu_pdata_arrays &pdata, const gpu_boxsize &box, int Nx, int Ny, int Nz, int order, float kappa, float rcut_ewald)
{
  //CUDA:
  cudaMalloc((void**)&(es_data.GPU_rho_real_space), sizeof(cufftComplex)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_green_hat), sizeof(float)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_k_vec), sizeof(float3)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_E_x), sizeof(cufftComplex)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_E_y), sizeof(cufftComplex)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_E_z), sizeof(cufftComplex)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.GPU_field), sizeof(float3)*Nx*Ny*Nz); 
  cudaMalloc((void**)&(es_data.vg), sizeof(float3)*Nx*Ny*Nz);
  cudaMalloc((void**)&(es_data.cuda_thermo_quantities), sizeof(float3));
  
  
  es_data.CPU_rho_coeff = (float*)malloc(order * (2*order+1) * sizeof(float));
  compute_rho_coeff(order, es_data.CPU_rho_coeff);
  cudaMemcpyToSymbol(weight_factors, &(es_data.CPU_rho_coeff[0]), order * (2*order+1) * sizeof(float));
  
  cufftPlan3d(&es_data.plan, Nx, Ny, Nz, CUFFT_C2C);
  
  
  //copy information to CPU here (stupid way, but works);
  struct gpu_boxsize CPU_box;
  struct gpu_boxsize *GPU_COPY_BOX;
  cudaMalloc((void**)&GPU_COPY_BOX, sizeof(struct gpu_boxsize));

  copy_data_kernel <<< 1,1 >>> (box, GPU_COPY_BOX);
      
  cudaMemcpy(&CPU_box, GPU_COPY_BOX, sizeof(struct gpu_boxsize), cudaMemcpyDeviceToHost);
      
  cudaFree(GPU_COPY_BOX);
      
  /* set up for a rectangular box */
  tensor_t lattice_vectors;
  lattice_vectors.xx = CPU_box.Lx;
  lattice_vectors.yx = 0.0;
  lattice_vectors.zx = 0.0;

  lattice_vectors.xy = 0.0;
  lattice_vectors.yy = CPU_box.Ly;
  lattice_vectors.zy = 0.0;

  lattice_vectors.xz = 0.0;
  lattice_vectors.yz = 0.0;
  lattice_vectors.zz = CPU_box.Lz;
       
  tensor_t inverse_lattice_vectors;
  inverse(&inverse_lattice_vectors, &lattice_vectors);
   
  tensor_t reciprocal_lattice_vectors;
  transpose(&reciprocal_lattice_vectors, &inverse_lattice_vectors);
  scalar_multiply(&reciprocal_lattice_vectors, 2*M_PI);
  
  float3* kvec_array = (float3*)malloc(Nx * Ny * Nz * sizeof(float3)); 
  int ix, iy, iz, kper, lper, mper, k, l, m;
   
  for (ix = 0; ix < Nx; ix++) {
    struct coord j;
    j.x = ix > Nx/2 ? ix - Nx : ix;
    for (iy = 0; iy < Ny; iy++) {
      j.y = iy > Ny/2 ? iy - Ny : iy;
      for (iz = 0; iz < Nz; iz++) {
	j.z = iz > Nz/2 ? iz - Nz : iz;
	float3 kvec;
	multiply(&kvec, &reciprocal_lattice_vectors, &j);
	kvec_array[iz + Nz * (iy + Ny * ix)] = kvec;
      }
    }
  }
     
  float3* cpu_vg = (float3 *)malloc(sizeof(float3)*Nx*Ny*Nz); 

  for(int x = 0; x < Nx; x++)
    {
      for(int y = 0; y < Ny; y++)
	{
	  for(int z = 0; z < Nz; z++)
	    {
	      float3 kvec = kvec_array[z + Nz * (y + Ny * x)];
	      float sqk =  kvec.x*kvec.x;
	      sqk += kvec.y*kvec.y;
	      sqk += kvec.z*kvec.z;
	
	      if (sqk == 0.0) 
		{
		  cpu_vg[z + Nz * (y + Ny * x)].x = 0.0f;
		  cpu_vg[z + Nz * (y + Ny * x)].y = 0.0f;
		  cpu_vg[z + Nz * (y + Ny * x)].z = 0.0f;
		}
	      else
		{
		  float vterm = -2.0 * (1.0/sqk + 0.25/(kappa*kappa));
		  cpu_vg[z + Nz * (y + Ny * x)].x =  1.0 + vterm*kvec.x*kvec.x;
		  cpu_vg[z + Nz * (y + Ny * x)].y =  1.0 + vterm*kvec.y*kvec.y;
		  cpu_vg[z + Nz * (y + Ny * x)].z =  1.0 + vterm*kvec.z*kvec.z;
		}
	    } 
	} 
    }
    
  float* green_hat = (float*)malloc(Nx * Ny * Nz * sizeof(float)); 
   
  int assignment_order = order;
  float snx, sny, snz, snx2, sny2, snz2;
  float argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
  float sum1, dot1, dot2;
  float numerator, denominator, sqk;

  float unitkx = (2.0*M_PI/CPU_box.Lx);
  float unitky = (2.0*M_PI/CPU_box.Ly);
  float unitkz = (2.0*M_PI/CPU_box.Lz);
   
    
  float xprd = CPU_box.Lx; 
  float yprd = CPU_box.Ly; 
  float zprd_slab = CPU_box.Lz; 
    
  float form = 1.0;
	
  gf_b = (float *)malloc(assignment_order*sizeof(float)); 
  compute_gf_denom(assignment_order);
	
#define EPS_HOC 1.0e-7

  float temp = floor(((kappa*xprd/(M_PI*Nx)) * 
		      pow(-log(EPS_HOC),0.25)));
  int nbx = (int)temp;

  temp = floor(((kappa*yprd/(M_PI*Nx)) * 
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
      
	sqk = pow(unitkx*kper,2.0f) + pow(unitky*lper,2.0f) + 
	  pow(unitkz*mper,2.0f);

	if (sqk != 0.0) {
	  numerator = form*12.5663706/sqk;
	  denominator = gf_denom(snx2,sny2,snz2,assignment_order);  
	  sum1 = 0.0;
	  for (ix = -nbx; ix <= nbx; ix++) {
	    qx = unitkx*(kper+(float)(Nx*ix));
	    sx = exp(-.25*pow(qx/kappa,2.0f));
	    wx = 1.0;
	    argx = 0.5*qx*xprd/(float)Nx;
	    if (argx != 0.0) wx = pow(sin(argx)/argx,assignment_order);
	    for (iy = -nby; iy <= nby; iy++) {
	      qy = unitky*(lper+(float)(Ny*iy));
	      sy = exp(-.25*pow(qy/kappa,2.0f));
	      wy = 1.0;
	      argy = 0.5*qy*yprd/(float)Ny;
	      if (argy != 0.0) wy = pow(sin(argy)/argy,assignment_order);
	      for (iz = -nbz; iz <= nbz; iz++) {
		qz = unitkz*(mper+(float)(Nz*iz));
		sz = exp(-.25*pow(qz/kappa,2.0f));
		wz = 1.0;
		argz = 0.5*qz*zprd_slab/(float)Nz;
		if (argz != 0.0) wz = pow(sin(argz)/argz,assignment_order);

		dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
		dot2 = qx*qx+qy*qy+qz*qz;
		sum1 += (dot1/dot2) * sx*sy*sz * pow(wx*wy*wz,2.0f);
	      }
	    }
	  }
	  green_hat[m + Nz * (l + Ny * k)] = numerator*sum1/denominator;
	} else green_hat[m + Nz * (l + Ny * k)] = 0.0;
      }
    }
  }
   
 
  cudaMemcpy(es_data.GPU_green_hat, green_hat, Nx * Ny * Nz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(es_data.GPU_k_vec, kvec_array, Nx * Ny * Nz * sizeof(float3), cudaMemcpyHostToDevice);  

  cudaMemcpy(es_data.vg, cpu_vg, Nx * Ny * Nz * sizeof(float3), cudaMemcpyHostToDevice);  
  free(cpu_vg);
  free(gf_b);
  free(green_hat);
  free(kvec_array);
  
  float scale = 1.0f/((float)(Nx * Ny * Nz));
  es_data.CPU_energy_virial_factor = 0.5 * CPU_box.Lx * CPU_box.Ly * CPU_box.Lz * scale * scale;
}


__device__ inline void atomicFloatAdd(float* address, float value)
{
  float old = value;  
  float new_old;

  do
    {
      new_old = atomicExch(address, 0.0f);
      new_old += old;
    }
  while ((old = atomicExch(address, new_old))!=0.0f);
}


__global__ void combined_green_e_kernel(cufftComplex* E_x, cufftComplex* E_y, cufftComplex* E_z, float3* k_vec, cufftComplex* rho, int Nx, int Ny, int Nz, float* green_function)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(tid < Nx * Ny * Nz)
    {
      float3 k_vec_local = k_vec[tid];
      cufftComplex E_x_local, E_y_local, E_z_local;
      float scale_times_green = green_function[tid] / ((float)(Nx*Ny*Nz));
      cufftComplex rho_local = rho[tid];
    
      rho[tid] = make_float2(0.0f,0.0f);
    
      rho_local.x *= scale_times_green;
      rho_local.y *= scale_times_green;
      
      E_x_local.x = k_vec_local.x * rho_local.y;
      E_x_local.y = -k_vec_local.x * rho_local.x;
    
      E_y_local.x = k_vec_local.y * rho_local.y;
      E_y_local.y = -k_vec_local.y * rho_local.x;
    
      E_z_local.x = k_vec_local.z * rho_local.y;
      E_z_local.y = -k_vec_local.z * rho_local.x;
    
    
      E_x[tid] = E_x_local;
      E_y[tid] = E_y_local;
      E_z[tid] = E_z_local;   
    }
}


__global__ void set_to_zero(cufftComplex* array, int Nx, int Ny, int Nz)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
 
  if(tid < Nx * Ny * Nz)
    {
      cufftComplex Zero = make_float2(0.0f,0.0f);
      array[tid] = Zero;
    }
}

__device__ inline void AddToGridpoint(int X, int Y, int Z, cufftComplex* array, float value, int Ny, int Nz)
{
  atomicFloatAdd(&array[Z + Nz * (Y + Ny * X)].x, value);
}

__device__ void compute_rho1d(float* rho1d, float dx, float dy, float dz, int interpolation_order)
{
  int k, l, k_order;
  int order = interpolation_order;

  k_order = -((1-order)/2);

    
  for (k = (1-order)/2; k <= order/2; k++) {
    rho1d[k+k_order + 0 * MaxOrder] = 0.0;
    rho1d[k+k_order + 1 * MaxOrder] = 0.0;
    rho1d[k+k_order + 2 * MaxOrder] = 0.0;
    for (l = order-1; l >= 0; l--) {
      rho1d[k+k_order + 0 * MaxOrder] = weight_factors[k+k_order + l*(2*order +1)] + rho1d[k+k_order + 0 * MaxOrder]*dx;
      rho1d[k+k_order + 1 * MaxOrder] = weight_factors[k+k_order + l*(2*order +1)] + rho1d[k+k_order + 1 * MaxOrder]*dy;
      rho1d[k+k_order + 2 * MaxOrder] = weight_factors[k+k_order + l*(2*order +1)] + rho1d[k+k_order + 2 * MaxOrder]*dz;
    }
  }
    
}

__device__ float get_rho1d(float delta, int interpolation_order, int y)
{
  int l, order = interpolation_order;

  float result;
    
  result = 0.0f;
  for (l = order-1; l >= 0; l--) {
    result = weight_factors[y + l*(2*order +1)] + result * delta;
  }

  return result;
}


__global__ void assign_charges_to_grid_kernel(gpu_pdata_arrays pdata, gpu_boxsize box, cufftComplex *rho_real_space, int Nx, int Ny, int Nz, int order)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < pdata.N)
    {
      //get particle information
      float qi = tex1Dfetch(pdata_charge_tex, idx);
      if(qi != 0.0) {
	float4 posi = tex1Dfetch(pdata_pos_tex, idx);
	//calculate dx, dy, dz for the charge density grid:
	float box_dx = box.Lx / ((float)Nx);
	float box_dy = box.Ly / ((float)Ny);
	float box_dz = box.Lz / ((float)Nz);
    
        
	//normalize position to gridsize:
	posi.x += box.Lx / 2.0f;
	posi.y += box.Ly / 2.0f;
	posi.z += box.Lz / 2.0f;
   
	posi.x /= box_dx;
	posi.y /= box_dy;
	posi.z /= box_dz;
    
    
	float shift, shiftone, x0, y0, z0, dx, dy, dz;
	int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
	nlower = -(order-1)/2;
	nupper = order/2;
    
	if (order % 2) 
	  {
	    shift =0.5;
	    shiftone = 0.0;
	  }
	else 
	  {
	    shift = 0.0;
	    shiftone = 0.5;
	  }
        
    
	nxi = __float2int_rd(posi.x + shift);
	nyi = __float2int_rd(posi.y + shift);
	nzi = __float2int_rd(posi.z + shift);
    
	dx = shiftone+(float)nxi-posi.x;
	dy = shiftone+(float)nyi-posi.y;
	dz = shiftone+(float)nzi-posi.z;
    
	int n,m,l;
    
	x0 = qi / (box_dx*box_dy*box_dz);
	for (n = nlower; n <= nupper; n++) {
	  mx = n+nxi;
	  if(mx >= Nx) mx -= Nx;
	  if(mx < 0)  mx += Nx;
	  y0 = x0*get_rho1d(dx, order, n-nlower);
	  for (m = nlower; m <= nupper; m++) {
	    my = m+nyi;
	    if(my >= Ny) my -= Ny;
	    if(my < 0)  my += Ny;
	    z0 = y0*get_rho1d(dy, order, m-nlower);
	    for (l = nlower; l <= nupper; l++) {
	      mz = l+nzi;
	      if(mz >= Nz) mz -= Nz;
	      if(mz < 0)  mz += Nz;
	      AddToGridpoint(mx, my, mz, rho_real_space, z0*get_rho1d(dz, order, l-nlower), Ny, Nz);
	    }
	  }
	}
      }
    }
}

__global__ void set_gpu_field_kernel(cufftComplex* E_x, cufftComplex* E_y, cufftComplex* E_z, float3* Electric_field, int Nx, int Ny, int Nz)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < Nx * Ny * Nz)
    {
      float3 local_field;
      local_field.x = E_x[tid].x;
      local_field.y = E_y[tid].x;
      local_field.z = E_z[tid].x;
      
      Electric_field[tid] = local_field;
    }
}

__global__ void calculate_forces_kernel(gpu_force_data_arrays force_data, gpu_pdata_arrays pdata, gpu_boxsize box, float3* Electric_field, int Nx, int Ny, int Nz, int order)
{  
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < pdata.N)
    {
      //get particle information
      float qi = tex1Dfetch(pdata_charge_tex, idx);
      if(qi != 0.0) {
	float4 posi = tex1Dfetch(pdata_pos_tex, idx);
    
	//calculate dx, dy, dz for the charge density grid:
	float box_dx = box.Lx / ((float)Nx);
	float box_dy = box.Ly / ((float)Ny);
	float box_dz = box.Lz / ((float)Nz);
    
	//normalize position to gridsize:
	posi.x += box.Lx / 2.0f;
	posi.y += box.Ly / 2.0f;
	posi.z += box.Lz / 2.0f;
   
	posi.x /= box_dx;
	posi.y /= box_dy;
	posi.z /= box_dz;
    
	float shift, shiftone, x0, y0, z0, dx, dy, dz;
	int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
	nlower = -(order-1)/2;
	nupper = order/2;
    
	float4 local_force = force_data.force[idx];

	if(order % 2) 
	  {
	    shift =0.5;
	    shiftone = 0.0;
	  }
	else 
	  {
	    shift = 0.0;
	    shiftone = 0.5;
	  }
    
    
	nxi = __float2int_rd(posi.x + shift);
	nyi = __float2int_rd(posi.y + shift);
	nzi = __float2int_rd(posi.z + shift);
    
	dx = shiftone+(float)nxi-posi.x;
	dy = shiftone+(float)nyi-posi.y;
	dz = shiftone+(float)nzi-posi.z;
 	int n,m,l;
    
  	for (n = nlower; n <= nupper; n++) {
	  mx = n+nxi;
	  if(mx >= Nx) mx -= Nx;
	  if(mx < 0)  mx += Nx;
	  x0 = get_rho1d(dx, order, n-nlower);
	  for (m = nlower; m <= nupper; m++) {
	    my = m+nyi;
	    if(my >= Ny) my -= Ny;
	    if(my < 0)  my += Ny;
	    y0 = x0*get_rho1d(dy, order, m-nlower);
	    for (l = nlower; l <= nupper; l++) {
	      mz = l+nzi;
	      if(mz >= Nz) mz -= Nz;
	      if(mz < 0)  mz += Nz;
	      z0 = y0*get_rho1d(dz, order, l-nlower);
	      float3 local_field = Electric_field[mz + Nz * (my + Ny * mx)];
	      local_force.x += qi*z0*local_field.x;
	      local_force.y += qi*z0*local_field.y;
	      local_force.z += qi*z0*local_field.z;
	    }
	  }
	}
    
	force_data.force[idx] = local_force;
      }
    }
} 

__global__ void calculate_thermo_quantities_kernel(cufftComplex* rho, float* green_function, float3* GPU_virial_energy, float3* vg, int Nx, int Ny, int Nz)
{
  int threadx = blockIdx.x * blockDim.x + threadIdx.x;
  int thready = blockIdx.y * blockDim.y + threadIdx.y;

  if((threadx < Nx) && (thready < Ny))
    {
      float2 local_GPU_virial_energy = make_float2(0.0f,0.0f);
      float3 local_vg;
      float local_green, green_times_rho_square;
      cufftComplex rho_local;
      for(int z = 0; z < Nz; z++)
	{
	  local_vg = vg[z + Nz * (thready + Ny * threadx)];
	  local_green = green_function[z + Nz * (thready + Ny * threadx)];
	  rho_local = rho[z + Nz * (thready + Ny * threadx)];
	
	  green_times_rho_square = local_green * (rho_local.x * rho_local.x + rho_local.y * rho_local.y);
	  local_GPU_virial_energy.x += green_times_rho_square * (local_vg.x + local_vg.y + local_vg.z);
	  local_GPU_virial_energy.y += green_times_rho_square ;
	}
  
      atomicFloatAdd(&GPU_virial_energy[0].x, local_GPU_virial_energy.x);
      atomicFloatAdd(&GPU_virial_energy[0].y, local_GPU_virial_energy.y);

    }
}

__global__ void get_charge(gpu_pdata_arrays pdata, float3 *GPU_virial_energy)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < pdata.N) {
    float qi = tex1Dfetch(pdata_charge_tex, idx);
    atomicFloatAdd(&GPU_virial_energy[0].x, qi);
  }
}

__global__ void get_charge_squared(gpu_pdata_arrays pdata, float3 *GPU_virial_energy)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < pdata.N) {
    float qi = tex1Dfetch(pdata_charge_tex, idx);
    atomicFloatAdd(&GPU_virial_energy[0].z, qi*qi);
  }

}

float3 calculate_thermo_quantities(const gpu_pdata_arrays &pdata, const gpu_boxsize &box)
{
  if(es_data.electrostatics_allocation_bool)
    {
      //kernel calling parameters for all grid dependent kernels
      int new_blockzise = 512;
      int new_gridsize = es_data.Nx*es_data.Ny*es_data.Nz / new_blockzise + 1;
      
      // setup the grid to run the kernel
      int blocksize = 512;
      dim3 grid( pdata.N / blocksize + 1, 1, 1);
      dim3 threads(blocksize, 1, 1);

      // bind the position texture
      pdata_pos_tex.normalized = false;
      pdata_pos_tex.filterMode = cudaFilterModePoint;
      cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);

      // bind the charge texture
      pdata_charge_tex.normalized = false;
      pdata_charge_tex.filterMode = cudaFilterModePoint;
      error = cudaBindTexture(0, pdata_charge_tex, pdata.charge, sizeof(float) * pdata.N);
	  
      //assign the charge density to the gridpoints
      assign_charges_to_grid_kernel <<< grid, threads >>> (pdata, box, es_data.GPU_rho_real_space, es_data.Nx, es_data.Ny, es_data.Nz, es_data.interpolation_order);
      cudaThreadSynchronize();    
    
      //call the forward FFT for the charge density
      cufftExecC2C(es_data.plan, es_data.GPU_rho_real_space, es_data.GPU_rho_real_space, CUFFT_FORWARD);
      cudaThreadSynchronize();
	  
      //calculate the virial and energy:  
      float3 CPU_virial_energy = make_float3(0.0f, 0.0f, 0.0f);
      cudaMemcpy(es_data.cuda_thermo_quantities, &CPU_virial_energy, sizeof(float3), cudaMemcpyHostToDevice);  
      get_charge_squared <<< grid, threads >>> (pdata, es_data.cuda_thermo_quantities);
     
      dim3 thermo_block(8,8,1);
      dim3 thermo_grid(es_data.Nx/thermo_block.x, es_data.Ny/thermo_block.y, 1);
      calculate_thermo_quantities_kernel <<< thermo_grid, thermo_block >>> (es_data.GPU_rho_real_space, es_data.GPU_green_hat, es_data.cuda_thermo_quantities, es_data.vg, es_data.Nx, es_data.Ny, es_data.Nz);
	
      //copy to CPU:
      cudaMemcpy(&CPU_virial_energy, es_data.cuda_thermo_quantities, sizeof(float3), cudaMemcpyDeviceToHost);

      struct gpu_boxsize CPU_box;
      struct gpu_boxsize *GPU_COPY_BOX;
      cudaMalloc((void**)&GPU_COPY_BOX, sizeof(struct gpu_boxsize));

      copy_data_kernel <<< 1,1 >>> (box, GPU_COPY_BOX);
      
      cudaMemcpy(&CPU_box, GPU_COPY_BOX, sizeof(struct gpu_boxsize), cudaMemcpyDeviceToHost);
      
      cudaFree(GPU_COPY_BOX);
      
      CPU_virial_energy.x *= es_data.CPU_energy_virial_factor / (3.0f * CPU_box.Lx * CPU_box.Ly * CPU_box.Lz);
      CPU_virial_energy.y *= es_data.CPU_energy_virial_factor;
	
      CPU_virial_energy.y -= CPU_virial_energy.z * es_data.kappa / sqrt(M_PI);

      set_to_zero <<< new_gridsize , new_blockzise >>> (es_data.GPU_rho_real_space, es_data.Nx, es_data.Ny, es_data.Nz);
      cudaThreadSynchronize();  
      return CPU_virial_energy;
	
	
    }
  else 
    return make_float3(0.0f, 0.0f, 0.0f);
}


void electrostatics_calculation(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const float3 *d_params, const float *d_rcutsq)
{
  //first time allocation of memory-------------------------------------
  int blocksize = 512;
  dim3 grid( pdata.N / blocksize + 1, 1, 1);
  dim3 threads(blocksize, 1, 1);

  if(!es_data.electrostatics_allocation_bool)
    {
      float3 cpu_params, *GPU_charge;

      cudaMalloc((void**)&(GPU_charge), sizeof(float3));

      cudaMemcpy(&cpu_params, d_params, sizeof(float3), cudaMemcpyDeviceToHost);
      printf("kappa = %g grid = %d order = %d\n",cpu_params.x, (int)cpu_params.y, (int)cpu_params.z);

      //Store values:
      es_data.Nx = cpu_params.y;
      es_data.Ny = es_data.Nx;
      es_data.Nz = es_data.Nx;

      es_data.interpolation_order = (int)cpu_params.z;
      int interpolation_order = es_data.interpolation_order;
      es_data.show_virial_flag = 0;

      int N = es_data.Nx;

      es_data.electrostatics_allocation_bool = 1;
      if(!(N == 2)&& !(N == 4)&& !(N == 8)&& !(N == 16)&& !(N == 32)&& !(N == 64)&& !(N == 128)&& !(N == 256)&& !(N == 512)&& !(N == 1024))
	{
	  printf("\n\n ------ ATTENTION gridsize should be a power of 2 ------ \n\n");
	}
      if (interpolation_order * (2*interpolation_order +1) > CONSTANT_SIZE)
	{
	  printf("interpolation order too high, doesn't fit into constant array\n");
	  exit(1);
	}
      if (interpolation_order > MaxOrder)
	{
	  printf("interpolation order too high\n");
	  exit(1);
	}
            
      float cpu_rcutsq;
      cudaMemcpy(&cpu_rcutsq, d_rcutsq, sizeof(float), cudaMemcpyDeviceToHost);
      
      es_data.r_cutoff = cpu_rcutsq;
      es_data.kappa = cpu_params.x;
      
      electrostatics_allocation(pdata, box, es_data.Nx, es_data.Ny, es_data.Nz, es_data.interpolation_order, es_data.kappa, es_data.r_cutoff);

      // bind the charge texture
      pdata_charge_tex.normalized = false;
      pdata_charge_tex.filterMode = cudaFilterModePoint;
      cudaError_t error = cudaBindTexture(0, pdata_charge_tex, pdata.charge, sizeof(float) * pdata.N);
      struct gpu_boxsize CPU_box;
      struct gpu_boxsize *GPU_COPY_BOX;
      cudaMalloc((void**)&GPU_COPY_BOX, sizeof(struct gpu_boxsize));

      copy_data_kernel <<< 1,1 >>> (box, GPU_COPY_BOX);
      cudaMemcpy(&CPU_box, GPU_COPY_BOX, sizeof(struct gpu_boxsize), cudaMemcpyDeviceToHost);

      float3 CPU_charge = make_float3(0.0f, 0.0f, 0.0f);
      cudaMemcpy(GPU_charge, &CPU_charge, sizeof(float3), cudaMemcpyHostToDevice);  
      get_charge <<< grid, threads >>> (pdata, GPU_charge);
      get_charge_squared <<< grid, threads >>> (pdata, GPU_charge);
      cudaMemcpy(&CPU_charge, GPU_charge, sizeof(float3), cudaMemcpyDeviceToHost);

      float hx =  CPU_box.Lx/es_data.Nx;
      float hy =  CPU_box.Ly/es_data.Ny;
      float hz =  CPU_box.Lz/es_data.Nz;

      float lprx = rms(hx, CPU_box.Lx, pdata.N, interpolation_order, es_data.kappa, CPU_charge.z); 
      float lpry = rms(hy, CPU_box.Lz, pdata.N, interpolation_order, es_data.kappa, CPU_charge.z);
      float lprz = rms(hz, CPU_box.Lz, pdata.N, interpolation_order, es_data.kappa, CPU_charge.z);
      float lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
      float spr = 2.0*CPU_charge.z*exp(-es_data.kappa*es_data.kappa*cpu_rcutsq) / sqrt(pdata.N*sqrt(cpu_rcutsq)*CPU_box.Lx*CPU_box.Ly*CPU_box.Lz);
     
      double RMS_error = MAX(lpr,spr);
      if(RMS_error > 0.1) {
	printf("!!!!!!!\n!!!!!!!\n!!!!!!!\nWARNING RMS error of %g is probably too high\n!!!!!!!\n!!!!!!!\n!!!!!!!\n", RMS_error);
      }
      else{
	printf("RMS error: %g\n", RMS_error);
      }
 
      if(CPU_charge.x > 0.0001 || CPU_charge.x < -0.0001) printf("WARNING system in not neutral, the net charge is %g\n", CPU_charge.x);
     
      
      printf("allocation for electrostatics done... \n");
      int new_blockzise = 512;
      int new_gridsize = es_data.Nx*es_data.Ny*es_data.Nz / new_blockzise + 1;
      //only for the first time needed, next time it is done in function new_combined_green_e_kernel
      set_to_zero <<< new_gridsize , new_blockzise >>> (es_data.GPU_rho_real_space, es_data.Nx, es_data.Ny, es_data.Nz);
      cudaThreadSynchronize();  
     
    }

  //kernel calling parameters for all grid dependent kernels
  int new_blockzise = 512;
  int new_gridsize = es_data.Nx*es_data.Ny*es_data.Nz / new_blockzise + 1;
    
  // setup the grid to run the particle kernel
    
    
  // bind the position texture
  pdata_pos_tex.normalized = false;
  pdata_pos_tex.filterMode = cudaFilterModePoint;
  cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);

  // bind the charge texture
  pdata_charge_tex.normalized = false;
  pdata_charge_tex.filterMode = cudaFilterModePoint;
  error = cudaBindTexture(0, pdata_charge_tex, pdata.charge, sizeof(float) * pdata.N);
     
  //assign the charge density to the gridpoints
  assign_charges_to_grid_kernel <<< grid, threads >>> (pdata, box, es_data.GPU_rho_real_space, es_data.Nx, es_data.Ny, es_data.Nz, es_data.interpolation_order);
  cudaThreadSynchronize();    
     
  //call the forward FFT for the charge density
  cufftExecC2C(es_data.plan, es_data.GPU_rho_real_space, es_data.GPU_rho_real_space, CUFFT_FORWARD);
  cudaThreadSynchronize();
    
  combined_green_e_kernel <<< new_gridsize, new_blockzise >>> (es_data.GPU_E_x, es_data.GPU_E_y, es_data.GPU_E_z, es_data.GPU_k_vec, es_data.GPU_rho_real_space,  es_data.Nx, es_data.Ny, es_data.Nz, es_data.GPU_green_hat);
       
  //backtransform field:
  cufftExecC2C(es_data.plan, es_data.GPU_E_x, es_data.GPU_E_x, CUFFT_INVERSE);
  cufftExecC2C(es_data.plan, es_data.GPU_E_y, es_data.GPU_E_y, CUFFT_INVERSE);
  cufftExecC2C(es_data.plan, es_data.GPU_E_z, es_data.GPU_E_z, CUFFT_INVERSE);
    
  //put field into float3 array
  set_gpu_field_kernel <<< new_gridsize, new_blockzise >>> (es_data.GPU_E_x, es_data.GPU_E_y, es_data.GPU_E_z, es_data.GPU_field, es_data.Nx, es_data.Ny, es_data.Nz);
  cudaThreadSynchronize();
  //calculate forces on particles:
  calculate_forces_kernel <<< grid, threads >>>(force_data, pdata, box, es_data.GPU_field, es_data.Nx, es_data.Ny, es_data.Nz, es_data.interpolation_order);
       
}
