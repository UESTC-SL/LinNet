#pragma once
#ifndef LINNET_VARIABLES_INL
#define LINNET_VARIABLES_INL

#include "global.hpp"

#ifdef LINNET_USE_OPENCV
#include <opencv2\opencv.hpp>
#endif

namespace lin {

	Size::Size() : w(0), h(0) {}

	Size::Size(const Size& rhs) : w(rhs.w), h(rhs.h) {}

	Size::Size(Size&& rhs) : w(rhs.w), h(rhs.h) {
		rhs.clear();
	}

	Size& Size::operator =(Size rhs) {
		swap(*this, rhs);
		return *this;
	}

	bool Size::operator ==(Size rhs) {
		return (this->w == rhs.w) && (this->h == rhs.h);
	}

	bool Size::operator !=(Size rhs) {
		return !(*this == rhs);
	}

	Size::~Size() {
		clear();
	}

	Size::Size(int width, int height) : w(width), h(height) {
		assert(checknonzero());
	}

	Size::Size(id width, id height) : w((int)width), h((int)height) {
		assert(checknonzero());
	}

	inline void Size::clear() {
		resize(0, 0);
	}

	inline void Size::resize(int x, int y) {
		w = x;
		h = y;
		assert(checknonzero());
	}

	inline bool Size::checknonzero() {
		return w >= 0 && h >= 0;
	}

	inline bool Size::checkpositive() {
		return w > 0 && h > 0;
	}

}

#endif
