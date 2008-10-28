#ifndef SARUPRNG_H
#define SARUPRNG_H

/*
 * Copyright (c) 2008 Steve Worley < m a t h g e e k@(my last name).com >
 *
BSD license will go here when it's released..
 */

/*
  C++ Saru PRNG by Steve Worley.   This is version 0.9, August 21 2008.

  Saru is a 32 bit PRNG with period of 3666320093*2^32. It passes even
  the most stringent randomness test batteries, including DIEHARD and
  the much more comprehensive BigCrush battery. It is designed for
  evaluation efficiency, a (relatively) compact state to allow fast
  copying, and most notably the ability to very efficiently advance
  (or rewind) its state. This advancement feature is useful for game
  and graphics tools, and is especially attractive for multithreaded
  and GPU applications.
  
  The Saru generator is an original work and (deliberately)
  unpatented. See the algorithm description at
  worley.com/mathgeek/saru.html. Updates/fixes to this code (and a plain C
  implementation) can be found there too.

-----------------

Usage:

  Constructors for 0, 1, 2, or 3 integer seeds.
Saru z, s(12345), t(123, 456), u(123, 456, 789);

  Advance state by 1, and output a 32 bit integer pseudo-random value.
cout << s.u32() << endl; //  Passes BigCrush and DIEHARD

  Advance state by 1, and output a double precision [0..1) floating point 
cout << s.d() << endl;

  Advance state by 1, and output a single precision [0..1) floating point 
cout << s.f() << endl;

  Move the generator state forwards a variable number of steps
s.advance(steps);

  Efficient state advancement or rewind when delta is known at compiletime
s.advance<123>(); 
s.rewind<123>(); 

Advance state by a different step (other than 1) and output a pseudorandom value.
cout << s.u32<4>() << endl; // skips forward 4 values and outputs the prand.

  Small structure size (64 bits of plain old data) means it's easy and fast
  to copy, store, pass, or revert the state.
z=s;

  Fork the PRNG, creating a new independent stream, seeded using
  current generator's state. Template seeding allows multiple
  independent children forks. 
Saru n=s.fork<123>();


  In practice you will likely extend or wrap this class to provide
  more specific functionality such as different output distributions.

---------------------

*/
class Saru {
 public:

   Saru() {state.x=0x12345678; state.y=12345678;}
   Saru(unsigned int seed);
   Saru(unsigned int seed1, unsigned int seed2);
   Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3);

  /* Efficient compile-time computed advancements */
   template <unsigned int steps> __device__ inline  void advance() 
    { advanceWeyl<steps>(); advanceLCG<steps>(); }
   
   // OK to advance negative steps in LCG, it's done mod 2^32 so it's
   // the same as advancing 2^32-1 steps, which is correct!
   template <unsigned int steps> __device__ inline void rewind() 
     { rewindWeyl<steps>(); advanceLCG<-steps>(); }
   
   /* Slower (but still reasonable) run-time computed advancements */
   __device__ inline void advance(unsigned int steps);

   template <unsigned int seed> __device__ Saru fork() const; 
   
  template <unsigned int steps> __device__ inline unsigned int u32();
  template <unsigned int steps> __device__ inline float f();
  template <unsigned int steps> __device__ inline double d();
  template <unsigned int steps> __device__ inline float f(float low, float high);
  template <unsigned int steps> __device__ inline double d(double low, double high);
  __device__ inline unsigned int u32();
  __device__ inline float f();
  __device__ inline double d();
  __device__ inline float f(float low, float high);
  __device__ inline double d(double low, double high);

  uint2 state;
   
 private:
   
   /* compile-time metaprograms to compute LCG and Weyl advancement */
   
   
   /* Computes A^N mod 2^32 */
   template<unsigned int A, unsigned int N> struct CTpow 
   { static const unsigned int value=(N&1?A:1)*CTpow<A*A, N/2>::value; };
  /* Specialization to terminate recursion: A^0 = 1 */
   template<unsigned int A> struct CTpow<A, 0> { static const unsigned int value=1; };
   
   /* CTpowseries<A,N> computes 1+A+A^2+A^3+A^4+A^5..+A^(N-1) mod 2^32.
      We do NOT use the more elegant formula (a^N-1)/(a-1) (see Knuth
      3.2.1), because it's more awkward to compute with the CPU's
      inherent mod 2^32.
      
      Based on recursion:
      g(A,n)= (1+A)*g(A*A, n/2);      if n is even
      g(A,n)= 1+A*(1+A)*g(A*A, n/2);  if n is ODD (since n/2 truncates)   */
   template<unsigned int A, unsigned int N>
     struct CTpowseries {
       static const unsigned int recurse=(1+A)*CTpowseries<A*A, N/2>::value;
       static const unsigned int value=  (N&1) ? 1+A*recurse : recurse;
     };
   template<unsigned int A>
     struct CTpowseries<A, 0> { static const unsigned int value=0; };  
   template<unsigned int A>
     struct CTpowseries<A, 1> { static const unsigned int value=1; };
   
   
   /* Compute A*B mod m.  Tricky only because of implicit 2^32 modulus.
     Uses recursion. 
     if A is even, then A*B mod m =  (A/2)*(B+B mod m) mod m.
     if A is odd,  then A*B mod m =  (B+((A/2)*(B+B mod m) mod m)) mod m.  */
   template <unsigned int A, unsigned int B, unsigned int m>
    struct CTmultmod {
      static const unsigned int temp= // (A/2)*(B*2) mod m
      CTmultmod< A/2, (B>=m-B ? B+B-m : B+B), m>::value;
      static const unsigned int value= A&1 ? ((B>=m-temp) ? B+temp-m: B+temp) : temp;
    };
   template <unsigned int B, unsigned int m>  /* terminate the recursion */
     struct CTmultmod<0, B, m> { static const unsigned int value=0; };
   
   template <unsigned int offset, unsigned int delta, 
     unsigned int modulus, unsigned int steps> 
     __device__ inline unsigned int advanceAnyWeyl(unsigned int);

   
   static const unsigned int LCGA=0x4beb5d59; // Full period 32 bit LCG
   static const unsigned int LCGC=0x2600e1f7; 
   static const unsigned int oWeylPeriod=0xda879add; // Prime period 3666320093
   static const unsigned int oWeylOffset=0x8009d14b;
   static const unsigned int oWeylDelta=(oWeylPeriod-0x80000000)+(oWeylOffset-0x80000000); // wraps mod 2^32
   
   /* Compile-time template function to efficently advance a state x with
      a LCG (mod 2^32) N steps.  Runtime, this all becomes a super-simple
      single multiply and add. */ 
   template <unsigned int steps> __device__ inline  void advanceLCG()
     { state.x=CTpow<LCGA, steps>::value*state.x+LCGC*CTpowseries<LCGA, steps>::value; }
   
   template <unsigned int steps> __device__ inline  void advanceWeyl()
     { state.y=advanceAnyWeyl<oWeylOffset, oWeylDelta, oWeylPeriod, steps>(state.y); }
   
   template <unsigned int steps> __device__ inline  void rewindWeyl()
     { state.y=advanceAnyWeyl<oWeylOffset, oWeylPeriod-oWeylDelta, 
	 oWeylPeriod, steps>(state.y); }
      
};

// partial specialization to make a special case for step of 1
template <> __device__ inline void  Saru::advanceWeyl<1>() /* especially efficient single step  */
    { state.y=state.y+oWeylOffset+((((signed int)state.y)>>31)&oWeylPeriod); }


/* This seeding was carefully tested for good churning with 1, 2, and
   3 bit flips.  All 32 incrementing counters (each of the circular
   shifts) pass the TestU01 Crush tests. */
Saru::Saru(unsigned int seed) 
{

  /* Good seeding, but still placeholder beta as more hashes cook. */
  unsigned int A, B, x;
  x=seed;

  unsigned int C1=0x8c9db1b9;
  unsigned int C4=0x56dde7b5;
  unsigned int S0=11;
  unsigned int S1=10;
  unsigned int S2=2;
  unsigned int S3=9;
  unsigned int S4=19;
  
  A = C1*((x)^(x>>S4)); 
  B = (x) ^ (A>>S1); 
  A = (A^B)  + (B<<S2); 
  B = (B)  + (((signed long)A)>>S3); 
  A = A^(B<<S0); 
  B = 0x80100000+(C4>>2)+(B>>1);

  state.x=A;
  state.y=B;

}

/* seeding from 2 samples. We lose one bit of entropy since our input
   seeds have 64 bits but at the end, after mixing, we have just 63. */
Saru::Saru(unsigned int seed1, unsigned int seed2)
{
  /* Good seeding, but still placeholder beta as more hashes cook. */
  state.x = seed1+(seed2<<21);
  state.y = seed2+(((signed int)state.x)>>16);
  state.x = state.x ^ (state.y>>10);
  state.x = 0x1fc4ce47*(state.x^(state.x>>13)); 
  state.y = (state.y+0xcc00729f) ^ (state.x>>18); 
  state.x = (state.x^state.y) + (state.y<<15); 
  state.y = (state.x+state.y) + (((signed int)state.x)>>27); 
  state.x = state.x^(state.y<<3); 
  state.y = 0x856C4555+(state.y>>1);
}


/* 3 seeds. We have to premix the seeds before dropping to 64 bits.
   TODO: this may be better optimized in a future version */
Saru::Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3) 
{
  /* Good seeding, but still placeholder beta as more hashes cook. */
  seed1  = seed1+(seed2<<20);
  seed2  = seed2+(seed1<<9);
  seed3  = seed3+(seed2>>13);
  seed1  = seed1^(seed3<<3);
  seed2  = seed2+(seed1>>18);
  seed3  = seed3^(seed2<<10);
  /* Drop our entropy from 96 mixed bits to 64 */
  seed1  += 0xDEADBEEF&seed3;
  seed2  += (~0xDEADBEEF)&seed3;
  state.x = seed1+(seed2<<21);
  state.y = seed2+(((signed int)state.x)>>16);
  state.x = state.x ^ (state.y>>10);
  state.x = 0x1fc4ce47*(state.x^(state.x>>13)); 
  state.y = (state.y+0xcc00729f) ^ (state.x>>18); 
  state.x = (state.x^state.y) + (state.y<<15); 
  state.y = (state.x+state.y) + (((signed int)state.x)>>27); 
  state.x = state.x^(state.y<<3); 
  state.y = 0x856C4555+(state.y>>1);
}

template <unsigned int offset, unsigned int delta, 
	  unsigned int modulus, unsigned int steps> 
  inline unsigned int Saru::advanceAnyWeyl(unsigned int x) 
{
  const unsigned int fullDelta=CTmultmod<delta, steps%modulus, modulus>::value;
  /* the runtime code boils down to this single constant-filled line. */
  return x+((x-offset>modulus-fullDelta) ? fullDelta-modulus : fullDelta);
}

__device__ inline void Saru::advance(unsigned int steps)
{
  // Computes the LCG advancement AND the Weyl D*E mod m simultaneously

  unsigned int currentA=LCGA;
  unsigned int currentC=LCGC;

  unsigned int currentDelta=oWeylDelta;
  unsigned int netDelta=0;
  
  while (steps) {    
    if (steps&1) {
      state.x=currentA*state.x+currentC; // LCG step
      if (netDelta<oWeylPeriod-currentDelta) netDelta+=currentDelta; 
      else netDelta+=currentDelta-oWeylPeriod;
    }
    
    // Change the LCG to step at twice the rate as before
    currentC+=currentA*currentC;
    currentA*=currentA;          
    
    // Change the Weyl delta to step at 2X rate
    if (currentDelta<oWeylPeriod-currentDelta) currentDelta+=currentDelta;
    else currentDelta+=currentDelta-oWeylPeriod;
    
    steps/=2;
  }
  
  // Apply the net delta to the Weyl state.
  if (state.y-oWeylOffset<oWeylPeriod-netDelta) state.y+=netDelta; 
  else state.y+=netDelta-oWeylPeriod;
}


template <unsigned int seed> 
__device__ Saru Saru::fork() const
{
  const unsigned int churned1=0xDEADBEEF ^ (0x1fc4ce47*(seed^(seed>>13)));
  const unsigned int churned2=0x1234567+(0x82948463*(churned1^(churned1>>20)));
  const unsigned int churned3=0x87654321^(0x87655677*(churned2^(churned2>>16)));
  /* The above lines are FREE since they'll be precomputed at compile time. They
     take user values like 1 2 3 and hash them to become roughly uncorrelated. */
  Saru z;

  z.state.x=churned2+state.x+(churned3^state.y);
  unsigned int add=(z.state.y+churned1)>>1;
  if (z.state.y-oWeylOffset<oWeylPeriod-add) z.state.y+=add;
  else z.state.y+=add-oWeylPeriod;
  return z;
}

/* Core PRNG evaluation. Very simple! */
template <unsigned int steps>
__device__ inline unsigned int Saru::u32()
{
  advanceLCG<steps>();
  advanceWeyl<steps>();
  unsigned int v=(state.x ^ (state.x>>26))+state.y;
  return (v^(v>>20))*0x6957f5a7;
}

__device__ inline unsigned int Saru::u32()
{
  return u32<1>();
}


/* Floats have 23 bits of mantissa. We take 31 p-rand bits, cast to
   signed int and simply multiply to get the (0,1] range. We shift and cast
   to long to boost x86 conversion speed, see worley.com/mathgeek/floatconvert.html. */

template <unsigned int steps>
__device__ inline float Saru::f()
{
  return ((signed int)(u32<steps>()>>1))*(1.0f/0x80000000); 
}

/* for a range that doesn't start at 0, we use the full 32 bits since
   we need to add an offset anyway. We still use the long cast method. */
template <unsigned int steps>
__device__ inline float Saru::f(float low, float high)
{
  const float TWO_N32 = 0.232830643653869628906250e-9f; /* 2^-32 */
  return ((signed int)(u32<steps>()))*(TWO_N32*(high-low))+0.5f*(high+low);
}


__device__ inline float Saru::f()
{
  return f<1>();
}

__device__ inline float Saru::f(float low, float high)
{
  return f<1>(low, high);
}


/* Doubles have 52 bits of mantissa. Casting to a long allows faster
   conversion, even with the extra needed fp addition. We use less-random
   "state" bits for the lowest order bits simply for speed. Output is
   in the (0,1] range. See worley.com/mathgeek/floatconvert.html. */

template <unsigned int steps>
__device__ inline double Saru::d()
{
  const double TWO_N32 = 0.232830643653869628906250e-9; /* 2^-32 */
  signed int v=(signed int)u32<steps>(); // deliberate cast to signed int for conversion speed

  return (v*TWO_N32+(0.5+0.5*TWO_N32))+((long)state.x)*(TWO_N32*TWO_N32);
}

template <unsigned int steps>
__device__ inline double Saru::d(double low, double high)
{
  const double TWO_N32 = 0.232830643653869628906250e-9; /* 2^-32 */
  signed int v=(signed int)u32<steps>(); // deliberate cast to signed int for conversion speed
  return (v*TWO_N32*(high-low)+(high+low)*(0.5+0.5*TWO_N32))+
    ((long)state.x)*(TWO_N32*TWO_N32*(high-low));
}


__device__ inline double Saru::d()
{
  return d<1>();
}

__device__ inline double Saru::d(double low, double high)
{
  return d<1>(low, high);
}



#endif /* SARUPRNG_H */
