#pragma once
#ifndef LINNET_GLOBAL_HPP
#define LINNET_GLOBAL_HPP

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cassert>
#include <exception>
#include <stdexcept>


//----------------------------------------------------------
//Use OpenCV 3 to load and show images and for matrix representation.
//----------------------------------------------------------
#define LINNET_USE_OPENCV

#ifdef LINNET_USE_OPENCV
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>
#include <opencv2\videoio.hpp>
#endif

//----------------------------------------------------------
//Use OpenGL for visualization.
//*Not implemented yet.*
//----------------------------------------------------------
//#define LINNET_USE_OPENGL

//----------------------------------------------------------
//Use NVIDIA CUDA library for GPU acceleration.
//*Not implemented yet.*
//----------------------------------------------------------
//#define LINNET_USE_CUDA

//----------------------------------------------------------
//Use Boost::thread for multi-threading.
//*Not implemented yet.*
//----------------------------------------------------------
//#define LINNET_MULTI_THREAD

//----------------------------------------------------------
// Statements like:
//		#pragma message(Reminder "Fix this problem!")
// Which will cause messages like:
//		C:\Source\Project\main.cpp(47): Reminder: Fix this problem!
// to show up during compiles.  Note that you can NOT use the
// words "error" or "warning" in your reminders, since it will
// make the IDE think it should abort execution.  You can double
// click on these messages and jump to the line in question.
//----------------------------------------------------------
#define Stringize( L )			#L
#define MakeString( M, L )		M(L)
#define $Line					MakeString( Stringize, __LINE__ )
#define Reminder				__FILE__ "(" $Line ") : Reminder: "

#define Location				" at " + __FILE__ + " : Line " + std::to_string(__LINE__)

#define msgnote( S )		__pragma( message( Reminder S ) )

#define msgwarn( S )		std::cout << "Warning : " << (std::string)S + Location << std::endl

#define msgerror( S )		throw std::runtime_error( (std::string)(S) + Location )

#endif
