// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/Filesystem.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include "hoomd/Messenger.h"

using namespace std;
using namespace boost;

/*! \file test_messenger.cc
    \brief Unit test for Messenger
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE MessengerTests
#include "boost_utf_configure.h"

BOOST_AUTO_TEST_CASE( Messenger_basic )
    {
    Messenger msg;

    // check that the default streams are set as documented
    BOOST_CHECK_EQUAL(&(msg.error()), &cerr);
    BOOST_CHECK_EQUAL(&(msg.warning()), &cerr);
    BOOST_CHECK_EQUAL(msg.getNoticeLevel(), (unsigned int)2);
    BOOST_CHECK_EQUAL(&(msg.notice(0)), &cout);
    BOOST_CHECK_EQUAL(&(msg.notice(1)), &cout);
    BOOST_CHECK_EQUAL(&(msg.notice(2)), &cout);
    BOOST_CHECK_EQUAL(&(msg.notice(3)), &(msg.getNullStream()));
    cout << endl;

    msg.setNoticeLevel(10);
    BOOST_CHECK_EQUAL(msg.getNoticeLevel(), (unsigned int)10);
    }


BOOST_AUTO_TEST_CASE( Messenger_print )
    {
    Messenger msg;

    // try a few test messages (must be determined by hand runs)
    msg.error() << "This is an error message test " << 1 << " " << 786.345 << endl;
    msg.warning() << "This is a warning message test" << endl;
    msg.notice(9) << "This notice message should not be printed" << endl;
    msg.setNoticeLevel(10);
    msg.notice(9) << "And this one should" << endl;
    msg.notice(1) << "Notice level 1" << endl;
    }

BOOST_AUTO_TEST_CASE( Messenger_null )
    {
    Messenger msg;

    // try setting the null streams
    msg.error() << "This message should be printed" << endl;
    msg.setErrorStream(msg.getNullStream());
    msg.error() << "But this one should not" << endl;

    msg.warning() << "This message should be printed" << endl;
    msg.setWarningStream(msg.getNullStream());
    msg.warning() << "But this one should not" << endl;

    msg.notice(1) << "This message should be printed" << endl;
    msg.setNoticeStream(msg.getNullStream());
    msg.notice(1) << "But this one should not" << endl;
    }

BOOST_AUTO_TEST_CASE( Messenger_prefix )
    {
    Messenger msg;

    // check that set/get work on prefixes
    msg.setErrorPrefix("err");
    BOOST_CHECK_EQUAL(msg.getErrorPrefix(), string("err"));
    msg.setWarningPrefix("warn");
    BOOST_CHECK_EQUAL(msg.getWarningPrefix(), string("warn"));
    msg.setNoticePrefix("note");
    BOOST_CHECK_EQUAL(msg.getNoticePrefix(), string("note"));

    ostringstream strm;
    // check that the prefixes are used properly
    msg.setErrorStream(strm);
    msg.error() << "message";
    BOOST_CHECK_EQUAL(strm.str(), string("err: message"));

    strm.str("");
    msg.setWarningStream(strm);
    msg.warning() << "message";
    BOOST_CHECK_EQUAL(strm.str(), string("warn: message"));

    strm.str("");
    msg.setNoticeStream(strm);
    msg.notice(1) << "message";
    BOOST_CHECK_EQUAL(strm.str(), string("message"));

    strm.str("");
    msg.setNoticeStream(strm);
    msg.notice(5) << "message";
    BOOST_CHECK_EQUAL(strm.str(), string(""));
    msg.setNoticeLevel(5);
    msg.notice(5) << "message";
    BOOST_CHECK_EQUAL(strm.str(), string("note(5): message"));

    // try copying a messenger and make sure that it still works
    strm.str("");
    Messenger msg2;
    msg2 = msg;
    msg2.error() << "1" << endl;
    msg.warning() << "2" << endl; // intentional use of msg to see if it merges output as it should
    msg2.notice(1) << "3" << endl;
    msg2.notice(5) << "4" << endl;
    msg2.notice(6) << "5" << endl;
    BOOST_CHECK_EQUAL(strm.str(), string("err: 1\nwarn: 2\n3\nnote(5): 4\n"));
    }

BOOST_AUTO_TEST_CASE( Messenger_file )
    {
    // scope the messengers so that the file is closed and written
        {
        Messenger msg;

        msg.setErrorPrefix("err");
        msg.setWarningPrefix("warn");
        msg.setNoticePrefix("note");

        // test opening a file
        msg.openFile("test_messenger_output");

        // also test that we can write to the file from 2 messengers
        Messenger msg2(msg);
        msg2.setNoticeLevel(5);

        // write the file (also tests the *Str functions)
        msg.errorStr("Error 1\n");
        msg2.warningStr("Warning 2\n");
        msg.noticeStr(1, "Notice 3\n");
        msg.noticeStr(5, "This shouldn't be printed\n");
        msg2.noticeStr(5, "Notice 4\n");
        }

    // make sure the file was created
    BOOST_REQUIRE(filesystem::exists("test_messenger_output"));

    // read in the file and make sure correct data was written
    ifstream f("test_messenger_output");
    string line;

    getline(f, line);
    BOOST_CHECK_EQUAL(line, "err: Error 1");
    BOOST_REQUIRE(!f.bad());

    getline(f, line);
    BOOST_CHECK_EQUAL(line, "warn: Warning 2");
    BOOST_REQUIRE(!f.bad());

    getline(f, line);
    BOOST_CHECK_EQUAL(line, "Notice 3");
    BOOST_REQUIRE(!f.bad());

    getline(f, line);
    BOOST_CHECK_EQUAL(line, "note(5): Notice 4");
    BOOST_REQUIRE(!f.bad());
    f.close();

    unlink("test_messenger_output");
    }
