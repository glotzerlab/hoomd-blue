/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: imd.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2009/12/07 17:41:17 $ *
 ***************************************************************************
 * DESCRIPTION:
 *  Lowest level interactive MD communication routines.
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/
#include "imd.h"
#include "vmdsock.h"
#include <string.h>
#include <errno.h>
#include <stdlib.h>

/// IMD communication protocol message header structure
typedef struct {
  int32 type;
  int32 length;
} IMDheader;

#define HEADERSIZE 8
#define IMDVERSION 2

/* Only works with aligned 4-byte quantities, will cause a bus error */
/* on some platforms if used on unaligned data.                      */
void swap4_aligned(void *v, long ndata) {
  int *data = (int *) v;
  long i;
  int *N;
  for (i=0; i<ndata; i++) {
    N = data + i;
    *N=(((*N>>24)&0xff) | ((*N&0xff)<<24) |
        ((*N>>8)&0xff00) | ((*N&0xff00)<<8));
  }
}

static int32 imd_htonl(int32 h) {
  int32 n;
  ((char *)&n)[0] = (h >> 24) & 0x0FF;
  ((char *)&n)[1] = (h >> 16) & 0x0FF;
  ((char *)&n)[2] = (h >> 8) & 0x0FF;
  ((char *)&n)[3] = h & 0x0FF;
  return n;
}

/// structure used to perform byte swapping operations
typedef struct {
  unsigned int highest : 8;
  unsigned int high    : 8;
  unsigned int low     : 8;
  unsigned int lowest  : 8;
} netint;

static int32 imd_ntohl(int32 n) {
  int32 h = 0;
  netint net;

  memcpy((void *)&net,(void *)&n, sizeof(n));
  h |= net.highest << 24 | net.high << 16 | net.low << 8 | net.lowest;
  return h;
}

static void fill_header(IMDheader *header, IMDType type, int32 length) {
  header->type = imd_htonl((int32)type);
  header->length = imd_htonl(length);
}

static void swap_header(IMDheader *header) {
  header->type = imd_ntohl(header->type);
  header->length= imd_ntohl(header->length);
}

static int32 imd_readn(void *s, char *ptr, int32 n) {
  int32 nleft;
  int32 nread;

  nleft = n;
  while (nleft > 0) {
    if ((nread = vmdsock_read(s, ptr, nleft)) < 0) {
      if (errno == EINTR)
        nread = 0;         /* and call read() again */
      else
        return -1;
    } else if (nread == 0)
      break;               /* EOF */
    nleft -= nread;
    ptr += nread;
  }
  return n-nleft;
}

static int32 imd_writen(void *s, const char *ptr, int32 n) {
  int32 nleft;
  int32 nwritten;

  nleft = n;
  while (nleft > 0) {
    if ((nwritten = vmdsock_write(s, ptr, nleft)) <= 0) {
      if (errno == EINTR)
        nwritten = 0;
      else
        return -1;
    }
    nleft -= nwritten;
    ptr += nwritten;
  }
  return n;
}


int imd_disconnect(void *s) {
  IMDheader header;
  fill_header(&header, IMD_DISCONNECT, 0);
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}

int imd_pause(void *s) {
  IMDheader header;
  fill_header(&header, IMD_PAUSE, 0);
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}

int imd_kill(void *s) {
  IMDheader header;
  fill_header(&header, IMD_KILL, 0);
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}

static int imd_go(void *s) {
  IMDheader header;
  fill_header(&header, IMD_GO, 0);
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}


int imd_handshake(void *s) {
  IMDheader header;
  fill_header(&header, IMD_HANDSHAKE, 1);
  header.length = IMDVERSION;   /* Not byteswapped! */
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}

int imd_trate(void *s, int32 rate) {
  IMDheader header;
  fill_header(&header, IMD_TRATE, rate);
  return (imd_writen(s, (char *)&header, HEADERSIZE) != HEADERSIZE);
}

/* Data methods */
int imd_send_mdcomm(void *s,int32 n,const int32 *indices,const float *forces) {
  int rc;
  int32 size = HEADERSIZE+16*n;
  char *buf = (char *) malloc(sizeof(char) * size);
  fill_header((IMDheader *)buf, IMD_MDCOMM, n);
  memcpy(buf+HEADERSIZE, indices, 4*n);
  memcpy(buf+HEADERSIZE+4*n, forces, 12*n);
  rc = (imd_writen(s, buf, size) != size);
  free(buf);
  return rc;
}

int imd_send_energies(void *s, const IMDEnergies *energies) {
  int rc;
  int32 size = HEADERSIZE+sizeof(IMDEnergies);
  char *buf = (char *) malloc(sizeof(char) * size);
  fill_header((IMDheader *)buf, IMD_ENERGIES, 1);
  memcpy(buf+HEADERSIZE, energies, sizeof(IMDEnergies));
  rc = (imd_writen(s, buf, size) != size);
  free(buf);
  return rc;
}

int imd_send_fcoords(void *s, int32 n, const float *coords) {
  int rc;
  int32 size = HEADERSIZE+12*n;
  char *buf = (char *) malloc(sizeof(char) * size);
  fill_header((IMDheader *)buf, IMD_FCOORDS, n);
  memcpy(buf+HEADERSIZE, coords, 12*n);
  rc = (imd_writen(s, buf, size) != size);
  free(buf);
  return rc;
}

/* The IMD receive functions */
IMDType imd_recv_header_nolengthswap(void *s, int32 *length) {
  IMDheader header;
  if (imd_readn(s, (char *)&header, HEADERSIZE) != HEADERSIZE)
    return IMD_IOERROR;
  *length = header.length;
  swap_header(&header);
  return (IMDType) header.type;
}

IMDType imd_recv_header(void *s, int32 *length) {
  IMDheader header;
  if (imd_readn(s, (char *)&header, HEADERSIZE) != HEADERSIZE)
    return IMD_IOERROR;
  swap_header(&header);
  *length = header.length;
  return (IMDType) header.type;
}

int imd_recv_handshake(void *s) {
  int32 buf;
  IMDType type;

  /* Wait up to 5 seconds for the handshake to come */
  if (vmdsock_selread(s, 5) != 1) return -1;

  /* Check to see that a valid handshake was received */
  type = imd_recv_header_nolengthswap(s, &buf);
  if (type != IMD_HANDSHAKE) return -1;

  /* Check its endianness, as well as the IMD version. */
  if (buf == IMDVERSION) {
    if (!imd_go(s)) return 0;
    return -1;
  }

  swap4_aligned(&buf, 1);
  if (buf == IMDVERSION) {
    if (!imd_go(s)) return 1;
  }

  /* We failed to determine endianness. */
  return -1;
}

int imd_recv_mdcomm(void *s, int32 n, int32 *indices, float *forces) {
  if (imd_readn(s, (char *)indices, 4*n) != 4*n) return 1;
  if (imd_readn(s, (char *)forces, 12*n) != 12*n) return 1;
  return 0;
}

int imd_recv_energies(void *s, IMDEnergies *energies) {
  return (imd_readn(s, (char *)energies, sizeof(IMDEnergies))
          != sizeof(IMDEnergies));
}

int imd_recv_fcoords(void *s, int32 n, float *coords) {
  return (imd_readn(s, (char *)coords, 12*n) != 12*n);
}
