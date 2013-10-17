#ifndef SARUPRNGCPU_H
#define SARUPRNGCPU_H

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4307 )
#endif

/*
 * Copyright (c) 2008 Steve Worley < m a t h g e e k@(my last name).com >

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

  Saru() : state(0x12345678), wstate(12345678) {};
  inline Saru(unsigned int seed);
  inline Saru(unsigned int seed1, unsigned int seed2);
  inline Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3);


  /* Efficient compile-time computed advancements */
  template <unsigned int steps> inline void advance()
    { advanceWeyl<steps>(); advanceLCG<steps>(); }

  // OK to advance negative steps in LCG, it's done mod 2^32 so it's
  // the same as advancing 2^32-1 steps, which is correct!
  template <unsigned int steps> inline void rewind()
    { rewindWeyl<steps>(); advanceLCG<-steps>(); }

  /* Slower (but still reasonable) run-time computed advancement */
  inline void advance(unsigned int steps);

  void setstate(unsigned int istate, unsigned int iwstate)
    { state=istate; wstate=iwstate; }

  template <unsigned int seed> inline Saru fork() const;

  template <unsigned int steps> inline unsigned int u32();
  template <unsigned int steps> inline float f();
  template <unsigned int steps> inline double d();
  template <unsigned int steps> inline float f(float low, float high);
  template <unsigned int steps> inline double d(double low, double high);
  inline unsigned int u32();
  inline float f();
  inline double d();
  inline float f(float low, float high);
  inline double d(double low, double high);

  template<class Real> inline Real s();
  template<class Real> inline Real s(Real low, Real high);

 private:

  /* compile-time metaprograms to compute LCG and Weyl advancement */


  /* Computes A^N mod 2^32 */
  template<unsigned int A, unsigned int N> struct CTpow
  { static const unsigned int value=(N&1?A:1)*CTpow<A*A, N/2>::value; };
  /* Specialization to terminate recursion: A^0 = 1 */
  template<unsigned int A> struct CTpow<A, 0> { static const unsigned int value=1; };

  /* CTpowseries<A,N> computes 1+A+A^2+A^3+A^4+A^5..+A^(N-1) mod 2^32.
     We do NOT use the more elegant formula (a^N-1)/(a-1) (see Knuth
     3.2.1), because it's more awkward to compute with implicit mod 2^32.

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
     if A is even, then A*B mod m =   (A/2)*(B+B mod m) mod m.
     if A is odd,  then A*B mod m =  ((A/2)*(B+B mod m) mod m) + B mod m.  */
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
    inline unsigned int advanceAnyWeyl(unsigned int);

  static const unsigned int LCGA=0x4beb5d59; // Full period 32 bit LCG
  static const unsigned int LCGC=0x2600e1f7;
  static const unsigned int oWeylPeriod=0xda879add; // Prime period 3666320093
  static const unsigned int oWeylOffset=0x8009d14b;
  static const unsigned int oWeylDelta=oWeylPeriod+oWeylOffset; // wraps mod 2^32

/* Compile-time template function to efficently advance a state x with
   a LCG (mod 2^32) N steps.  Runtime, this all becomes a super-simple
   single multiply and add. */
  template <unsigned int steps> inline void advanceLCG()
    { state=CTpow<LCGA, steps>::value*state+LCGC*CTpowseries<LCGA, steps>::value; }

  template <unsigned int steps> inline void advanceWeyl()
    { wstate=advanceAnyWeyl<oWeylOffset, oWeylDelta, oWeylPeriod, steps>(wstate); }


  template <unsigned int steps> inline void rewindWeyl()
    { wstate=advanceAnyWeyl<oWeylOffset, oWeylPeriod-oWeylDelta,
    oWeylPeriod, steps>(wstate); }

  unsigned int state;  // LCG state
  unsigned int wstate; // Offset Weyl sequence state
};


// partial specialization to make a special case for step of 1
template <> inline void Saru::advanceWeyl<1>() /* especially efficient single step  */
{
  wstate=wstate+oWeylOffset+((((signed int)wstate)>>31)&oWeylPeriod);
}


/* This seeding was carefully tested for good churning with 1, 2, and
   3 bit flips.  All 32 incrementing counters (each of the circular
   shifts) pass the TestU01 Crush tests. */
inline Saru::Saru(unsigned int seed)
{
  state  = 0x79dedea3*(seed^(((signed int)seed)>>14));
  wstate = seed ^ (((signed int)state)>>8);
  state  = state + (wstate*(wstate^0xdddf97f5));
  wstate = 0xABCB96F7 + (wstate>>1);
}

/* seeding from 2 samples. We lose one bit of entropy since our input
   seeds have 64 bits but at the end, after mixing, we have just 63. */
inline Saru::Saru(unsigned int seed1, unsigned int seed2)
{
  seed2+=seed1<<16;
  seed1+=seed2<<11;
  seed2+=((signed int)seed1)>>7;
  seed1^=((signed int)seed2)>>3;
  seed2*=0xA5366B4D;
  seed2^=seed2>>10;
  seed2^=((signed int)seed2)>>19;
  seed1+=seed2^0x6d2d4e11;

  state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
  wstate = (state + seed2) ^ (((signed int)state)>>8);
  state  = state + (wstate*(wstate^0xdddf97f5));
  wstate = 0xABCB96F7 + (wstate>>1);

}


/* 3 seeds. We have to premix the seeds before dropping to 64 bits.
   TODO: this may be better optimized in a future version */
inline Saru::Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
  seed3^=(seed1<<7)^(seed2>>6);
  seed2+=(seed1>>4)^(seed3>>15);
  seed1^=(seed2<<9)+(seed3<<8);
  seed3^=0xA5366B4D*((seed2>>11) ^ (seed1<<1));
  seed2+=0x72BE1579*((seed1<<4)  ^ (seed3>>16));
  seed1^=0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
  seed2+=seed1*seed3;
  seed1+=seed3 ^ (seed2>>2);
  seed2^=((signed int)seed2)>>17;

  state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
  wstate = (state + seed2) ^ (((signed int)state)>>8);
  state  = state + (wstate*(wstate^0xdddf97f5));
  wstate = 0xABCB96F7 + (wstate>>1);
}


template <unsigned int offset, unsigned int delta,
      unsigned int modulus, unsigned int steps>
  inline unsigned int Saru::advanceAnyWeyl(unsigned int x)
{
  static const unsigned int fullDelta=CTmultmod<delta, steps%modulus, modulus>::value;
  /* runtime code boils down to this single constant-filled line. */
  return x+((x-offset>modulus-fullDelta) ? fullDelta-modulus : fullDelta);
}




inline void Saru::advance(unsigned int steps)
{
  // Computes the LCG advancement AND the Weyl D*E mod m simultaneously

  unsigned int currentA=LCGA;
  unsigned int currentC=LCGC;

  unsigned int currentDelta=oWeylDelta;
  unsigned int netDelta=0;

  while (steps) {
    if (steps&1) {
      state=currentA*state+currentC; // LCG step
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
  if (wstate-oWeylOffset<oWeylPeriod-netDelta) wstate+=netDelta;
  else wstate+=netDelta-oWeylPeriod;
}


template <unsigned int seed>
inline Saru Saru::fork() const
{
  static const unsigned int churned1=0xDEADBEEF ^ (0x1fc4ce47*(seed^(seed>>13)));
  static const unsigned int churned2=0x1234567+(0x82948463*(churned1^(churned1>>20)));
  static const unsigned int churned3=0x87654321^(0x87655677*(churned2^(churned2>>16)));
  /* The above lines are FREE since they'll be precomputed at compile time. They
     take user values like 1 2 3 and hash them to become roughly uncorrelated. */
  Saru z;

  z.state=churned2+state+(churned3^wstate);
  unsigned int add=(z.state+churned1)>>1;
  if (z.wstate-oWeylOffset<oWeylPeriod-add) z.wstate+=add;
  else z.wstate+=add-oWeylPeriod;
  return z;
}

/* Core PRNG evaluation. Very simple! */
template <unsigned int steps>
inline unsigned int Saru::u32()
{
  advanceLCG<steps>();
  advanceWeyl<steps>();
  unsigned int v=(state ^ (state>>26))+wstate;
  return (v^(v>>20))*0x6957f5a7;
}

inline unsigned int Saru::u32()
{
  return u32<1>();
}


/* Floats have 23 bits of mantissa. We take 31 p-rand bits, cast to
   signed int and simply multiply to get the (0,1] range. We shift and cast
   to long to boost x86 conversion speed, see worley.com/mathgeek/floatconvert.html. */

template <unsigned int steps>
inline float Saru::f()
{
  return ((signed int)(u32<steps>()>>1))*(1.0f/0x80000000);
}

/* for a range that doesn't start at 0, we use the full 32 bits since
   we need to add an offset anyway. We still use the long cast method. */
template <unsigned int steps>
inline float Saru::f(float low, float high)
{
  const float TWO_N32 = 0.232830643653869628906250e-9f; /* 2^-32 */
  return ((signed int)(u32<steps>()))*(TWO_N32*(high-low))+0.5f*(high+low);
}


inline float Saru::f()
{
  return f<1>();
}

inline float Saru::f(float low, float high)
{
  return f<1>(low, high);
}


/* Doubles have 52 bits of mantissa. Casting to a long allows faster
   conversion, even with the extra needed fp addition. We use less-random
   "state" bits for the lowest order bits simply for speed. Output is
   in the (0,1] range. See worley.com/mathgeek/floatconvert.html. */

template <unsigned int steps>
inline double Saru::d()
{
  const double TWO_N32 = 0.232830643653869628906250e-9; /* 2^-32 */
  signed int v=(signed int)u32<steps>(); // deliberate cast to signed int for conversion speed

  return (v*TWO_N32+(0.5+0.5*TWO_N32))+((long)state)*(TWO_N32*TWO_N32);
}

template <unsigned int steps>
inline double Saru::d(double low, double high)
{
  const double TWO_N32 = 0.232830643653869628906250e-9; /* 2^-32 */
  signed int v=(signed int)u32<steps>(); // deliberate cast to signed int for conversion speed
  return (v*TWO_N32*(high-low)+(high+low)*(0.5+0.5*TWO_N32))+
    ((long)state)*(TWO_N32*TWO_N32*(high-low));
}


inline double Saru::d()
{
  return d<1>();
}

inline double Saru::d(double low, double high)
{
  return d<1>(low, high);
}

template<class Real>
inline Real Saru::s()
    {
    // default implementation returns something ridiculous, so it is obvious when it is called
    return -1000000000;
    }

template<class Real>
inline Real Saru::s(Real low, Real high)
    {
    // default implementation returns something ridiculous, so it is obvious when it is called
    return -1000000000;
    }

template<>
inline float Saru::s()
    {
    return f();
    }

template<>
inline float Saru::s(float low, float high)
    {
    return f(low, high);
    }

template<>
inline double Saru::s()
    {
    return d();
    }

template<>
inline double Saru::s(double low, double high)
    {
    return d(low, high);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#endif /* SARUPRNGCPU_H */
