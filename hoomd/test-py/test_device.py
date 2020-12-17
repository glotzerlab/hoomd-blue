# -*- coding: iso-8859-1 -*-
# Maintainer: tommy-waltmann

from hoomd import *
import os
import unittest

context.initialize()


class test_devices(unittest.TestCase):
        
    def test_gpu(self):
        
        if hoomd.options.mode == 'cpu':
            assertEquals(hoomd.context.current.device.gpu_ids, [])
        
    def test_mode(self):
        
        assertEquals(hoomd.context.current.device.mode, hoomd.options.mode)
        
    def test_num_threads(self):
        
        # test default and setter
        assertEquals(hoomd.context.current.device.num_threads, 1)
        hoomd.context.current.device.num_threads = 4
        assertEquals(hoomd.context.current.device.num_threads, 4)
        
    def test_notice_level(self):
        
        # test default and setter
        assertEquals(hoomd.context.current.device.notice_level, 2)
        hoomd.context.current.device.notice_level = 4
        assertEquals(hoomd.context.current.device.notice_level, 4)
        
    def test_memory_traceback(self):
        
        # test default and setter
        assertEquals(hoomd.context.current.device.memory_traceback, False)
        hoomd.context.current.device.memory_traceback = True
        assertEquals(hoomd.context.current.device.memory_traceback, True)
        
    def test_gpu_error_checking(self):
        
        # test default and setter
        assertEquals(hoomd.context.current.device.gpu_error_checking, False)
        hoomd.context.current.device.gpu_error_checking = True  # this shouldn't actually change anything in cpu mode
        
        if hoomd.context.current.device.mode == 'cpu':
            assertEquals(hoomd.context.current.device.gpu_error_checking, False)
        else:
            assertEquals(hoomd.context.current.device.gpu_error_checking, True)
        
    def test_msg_file(self):
        
        fname = "filename.txt"
        
        assertEquals(hoomd.conetxt.current.device.msg_file, None)
        hoomd.context.current.device.msg_file = fname
        assertEquals(hoomd.context.current.device.msg_file, fname)
        
    def test_nondefault(self):

        msg_file_name = "msg_file.txt"

        with tempfile.TemporaryDirectory() as d: 
        
            filepath = d + "/" + msg_file_name
        
            # use all the optional arguments       
            cpu = device.CPU(nthreads=2, 
                             communicator=comm.Communicator(nrank=4), 
                             msg_file=filepath, 
                             notice_level=8)
        
            # test the non-default args
            assertEquals(cpu.msg_file, msg_file_name)
            assertEquals(cpu.notice_level, 8)
            assertEquals(cpu.num_threads, 2)
            assertEquals(gpu.comm.num_ranks, 4)
        
            # test that the new file exists and is nonempty
            assertTrue(os.isPath(filepath))
            assertTrue(os.path(filepath).st_size > 0)
