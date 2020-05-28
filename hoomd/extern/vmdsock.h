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
 *      $RCSfile: vmdsock.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $      $Date: 2009/12/07 17:36:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   socket interface layer, abstracts platform-dependent routines/APIs
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

#if defined(VMDSOCKINTERNAL)

#if !defined(_MSC_VER)
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <sys/file.h>
#endif

typedef struct {
  struct sockaddr_in addr; /* address of socket provided by bind() */
  int addrlen;             /* size of the addr struct */
  int sd;                  /* socket file descriptor */
} vmdsocket;

#endif /* VMDSOCKINTERNAL */

#ifdef __cplusplus
extern "C" {
#endif

int   vmdsock_init(void);
void *vmdsock_create(void);
int   vmdsock_bind(void *, int);
int   vmdsock_listen(void *);
void *vmdsock_accept(void *);  /* return new socket */
int   vmdsock_connect(void *, const char *, int);
int   vmdsock_write(void *, const void *, int);
int   vmdsock_read(void *, void *, int);
int   vmdsock_selread(void *, int);
int   vmdsock_selwrite(void *, int);
void  vmdsock_shutdown(void *);
void  vmdsock_destroy(void *);

#ifdef __cplusplus
}
#endif
