/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2003 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: vmdsock.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.1 $      $Date: 2003/09/12 18:30:46 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   socket interface layer, abstracts platform-dependent routines/APIs
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

