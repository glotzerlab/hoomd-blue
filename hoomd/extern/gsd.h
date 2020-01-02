// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the General Simulation Data (GSD) project, released under the BSD 2-Clause License.

#ifndef __GSD_H__
#define __GSD_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \file gsd.h
    \brief Declare GSD data types and C API
*/

//! Identifiers for the gsd data chunk element types
enum gsd_type
    {
    GSD_TYPE_UINT8=1,
    GSD_TYPE_UINT16,
    GSD_TYPE_UINT32,
    GSD_TYPE_UINT64,
    GSD_TYPE_INT8,
    GSD_TYPE_INT16,
    GSD_TYPE_INT32,
    GSD_TYPE_INT64,
    GSD_TYPE_FLOAT,
    GSD_TYPE_DOUBLE
    };

//! Flag for GSD file open options
enum gsd_open_flag
    {
    GSD_OPEN_READWRITE=1,
    GSD_OPEN_READONLY,
    GSD_OPEN_APPEND
    };

//! GSD file header
/*! The GSD file header.

    \warning All members are **read-only** to the caller.
*/
struct gsd_header
    {
    uint64_t magic;
    uint64_t index_location;
    uint64_t index_allocated_entries;
    uint64_t namelist_location;
    uint64_t namelist_allocated_entries;
    uint32_t schema_version;            //!< Schema version: 0xaaaabbbb => aaaa.bbbb
    uint32_t gsd_version;               //!< File format version: 0xaaaabbbb => aaaa.bbbb
    char application[64];               //!< Name of generating application
    char schema[64];                    //!< Name of data schema
    char reserved[80];
    };

//! Index entry
/*! An index entry for a single chunk of data.

    \warning All members are **read-only** to the caller.
*/
struct gsd_index_entry
    {
    uint64_t frame;     //!< Frame index of the chunk
    uint64_t N;         //!< Number of rows in the chunk
    int64_t location;
    uint32_t M;         //!< Number of columns in the chunk
    uint16_t id;
    uint8_t type;       //!< Data type of the chunk
    uint8_t flags;
    };

//! Namelist entry
/*! An entry in the list of data chunk names

    \warning All members are **read-only** to the caller.
*/
struct gsd_namelist_entry
    {
    char name[64];      //!< Entry name
    };

//! File handle
/*! A handle to an open GSD file.

    This handle is obtained when opening a GSD file and is passed into every method that operates on the file.

    \warning All members are **read-only** to the caller.
*/
struct gsd_handle
    {
    int fd;
    struct gsd_header header;           //!< GSD file header
    void *mapped_data;
    size_t append_index_size;
    struct gsd_index_entry *index;
    struct gsd_namelist_entry *namelist;
    uint64_t namelist_num_entries;
    uint64_t index_written_entries;
    uint64_t index_num_entries;
    uint64_t cur_frame;
    int64_t file_size;                  //!< File size (in bytes)
    enum gsd_open_flag open_flags;      //!< Flags passed to gsd_open()
    bool needs_sync; //!< Whether the handle requires an fsync call (new data was written)
    };

//! Specify a version
uint32_t gsd_make_version(unsigned int major, unsigned int minor);

//! Create a GSD file
int gsd_create(const char *fname, const char *application, const char *schema, uint32_t schema_version);

//! Create and open a GSD file
int gsd_create_and_open(struct gsd_handle* handle,
                        const char *fname,
                        const char *application,
                        const char *schema,
                        uint32_t schema_version,
                        const enum gsd_open_flag flags,
                        int exclusive_create);

//! Open a GSD file
int gsd_open(struct gsd_handle* handle, const char *fname, const enum gsd_open_flag flags);

//! Truncate a GSD file
int gsd_truncate(struct gsd_handle* handle);

//! Close a GSD file
int gsd_close(struct gsd_handle* handle);

//! Move on to the next frame
int gsd_end_frame(struct gsd_handle* handle);

//! Write a data chunk to the current frame
int gsd_write_chunk(struct gsd_handle* handle,
                    const char *name,
                    enum gsd_type type,
                    uint64_t N,
                    uint32_t M,
                    uint8_t flags,
                    const void *data);

//! Find a chunk in the GSD file
const struct gsd_index_entry* gsd_find_chunk(struct gsd_handle* handle, uint64_t frame, const char *name);

//! Read a chunk from the GSD file
int gsd_read_chunk(struct gsd_handle* handle, void* data, const struct gsd_index_entry* chunk);

//! Get the number of frames in the GSD file
uint64_t gsd_get_nframes(struct gsd_handle* handle);

//! Query size of a GSD type ID
size_t gsd_sizeof_type(enum gsd_type type);

//! Search for chunk names in a gsd file
const char *gsd_find_matching_chunk_name(struct gsd_handle* handle, const char* match, const char *prev);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef __GSD_H__
