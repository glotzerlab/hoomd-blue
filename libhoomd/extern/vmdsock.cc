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
 *      $RCSfile: vmdsock.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $      $Date: 2009/12/07 17:36:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Socket interface, abstracts machine dependent APIs/routines. 
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

#define VMDSOCKINTERNAL 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER) 
#include <winsock2.h>
typedef int socklen_t;
#else
#include <arpa/inet.h>
#include <fcntl.h>

#include <sys/types.h>
#include <unistd.h>   /* for Linux */
#include <sys/socket.h>
#include <netdb.h>
#endif

#include <errno.h>

#include "vmdsock.h"

int vmdsock_init(void) {
#if defined(_MSC_VER)
  int rc = 0;
  static int initialized=0;

  if (!initialized) {
    WSADATA wsdata;
    rc = WSAStartup(MAKEWORD(1,1), &wsdata);
    if (rc == 0)
      initialized = 1;
  }

  return rc;
#else   
  return 0;
#endif
}


void * vmdsock_create(void) {
  vmdsocket * s;

  s = (vmdsocket *) malloc(sizeof(vmdsocket));
  if (s != NULL)
    memset(s, 0, sizeof(vmdsocket)); 

  if ((s->sd = (int)socket(PF_INET, SOCK_STREAM, 0)) == -1) {
    printf("Failed to open socket.");
    free(s);
    return NULL;
  }

  return (void *) s;
}

int  vmdsock_connect(void *v, const char *host, int port) {
  vmdsocket *s = (vmdsocket *) v;
  char address[1030];
  struct hostent *h;

  h=gethostbyname(host);
  if (h == NULL) 
    return -1;
  sprintf(address, "%d.%d.%d.%d",
    (unsigned char) h->h_addr_list[0][0],
    (unsigned char) h->h_addr_list[0][1],
    (unsigned char) h->h_addr_list[0][2],
    (unsigned char) h->h_addr_list[0][3]);

  memset(&(s->addr), 0, sizeof(s->addr)); 
  s->addr.sin_family = PF_INET;
  s->addr.sin_addr.s_addr = inet_addr(address);
  s->addr.sin_port = htons(port);  

  return connect(s->sd, (struct sockaddr *) &s->addr, sizeof(s->addr)); 
}

int vmdsock_bind(void * v, int port) {
  vmdsocket *s = (vmdsocket *) v;
  memset(&(s->addr), 0, sizeof(s->addr)); 
  s->addr.sin_family = PF_INET;
  s->addr.sin_port = htons(port);

  return bind(s->sd, (struct sockaddr *) &s->addr, sizeof(s->addr));
}

int vmdsock_listen(void * v) {
  vmdsocket *s = (vmdsocket *) v;
  return listen(s->sd, 5);
}

void *vmdsock_accept(void * v) {
  int rc;
  vmdsocket *new_s = NULL, *s = (vmdsocket *) v;
#if defined(ARCH_AIX5) || defined(ARCH_AIX5_64) || defined(ARCH_AIX6_64)
  unsigned int len;
#elif defined(SOCKLEN_T)
  SOCKLEN_T len;
#elif defined(ARCH_LINUXALPHA)
  socklen_t len;
#else
  int len;
#endif

  len = sizeof(s->addr);
  rc = (int)accept(s->sd, (struct sockaddr *) &s->addr, (socklen_t *)&len);
  if (rc >= 0) {
    new_s = (vmdsocket *) malloc(sizeof(vmdsocket));
    if (new_s != NULL) {
      *new_s = *s;
      new_s->sd = rc;
    }
  }
  return (void *)new_s;
}

int  vmdsock_write(void * v, const void *buf, int len) {
  vmdsocket *s = (vmdsocket *) v;
#if defined(_MSC_VER)
  return send(s->sd, (const char*) buf, len, 0);  // windows lacks the write() call
#else
  return write(s->sd, buf, len);
#endif
}

int  vmdsock_read(void * v, void *buf, int len) {
  vmdsocket *s = (vmdsocket *) v;
#if defined(_MSC_VER)
  return recv(s->sd, (char*) buf, len, 0); // windows lacks the read() call
#else
  return read(s->sd, buf, len);
#endif

}

void vmdsock_shutdown(void *v) {
  vmdsocket * s = (vmdsocket *) v;
  if (s == NULL)
    return;

#if defined(_MSC_VER)
  shutdown(s->sd, SD_SEND);
#else
  shutdown(s->sd, 1);  /* complete sends and send FIN */
#endif
}

void vmdsock_destroy(void * v) {
  vmdsocket * s = (vmdsocket *) v;
  if (s == NULL)
    return;

#if defined(_MSC_VER)
  closesocket(s->sd);
#else
  close(s->sd);
#endif
  free(s);  
}

int vmdsock_selread(void *v, int sec) {
  vmdsocket *s = (vmdsocket *)v;
  fd_set rfd;
  struct timeval tv;
  int rc;
 
  FD_ZERO(&rfd);
  FD_SET(s->sd, &rfd);
  memset((void *)&tv, 0, sizeof(struct timeval));
  tv.tv_sec = sec;
  do {
    rc = select(s->sd+1, &rfd, NULL, NULL, &tv);
  } while (rc < 0 && errno == EINTR);
  return rc;

}
  
int vmdsock_selwrite(void *v, int sec) {
  vmdsocket *s = (vmdsocket *)v;
  fd_set wfd;
  struct timeval tv;
  int rc;
 
  FD_ZERO(&wfd);
  FD_SET(s->sd, &wfd);
  memset((void *)&tv, 0, sizeof(struct timeval));
  tv.tv_sec = sec;
  do {
    rc = select(s->sd + 1, NULL, &wfd, NULL, &tv);
  } while (rc < 0 && errno == EINTR);
  return rc;
}
