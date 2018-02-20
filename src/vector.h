/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdint>
#include <ostream>

#include "real.h"

namespace fasttext {

class Matrix;
class QMatrix;

class Vector {

  public:
    int64_t size_;
    real* data_;
    bool dataShared_;

    explicit Vector(int64_t);
    explicit Vector(int64_t m, real* data);
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    ~Vector();

    real* data() {
      return data_;
    }

    const real* data() const {
      return data_;
    }

    real& operator[](int64_t i) {
      return data_[i];
    }

    const real& operator[](int64_t i) const {
      return data_[i];
    }

    int64_t size() const {
      return size_;
    }

    void zero();
    void mul(real);
    real norm() const;
    void addVector(const Vector& source);
    void addVector(const Vector&, real);
    void addRow(const Matrix&, int64_t);
    void addRow(const QMatrix&, int64_t);
    void addRow(const Matrix&, int64_t, real);
    void mul(const QMatrix&, const Vector&);
    void mul(const Matrix&, const Vector&);
    int64_t argmax();
};

std::ostream& operator<<(std::ostream&, const Vector&);

}
