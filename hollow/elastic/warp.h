// Hard coded elastic warps

#include <hollow/petsc/config.h>
#include <geode/array/Array.h>
#include <geode/math/constants.h>
namespace hollow {

// Since PetIGA currently has poor boundary condition support,
// we have to bake initial guess and boundary conditions into code.

template<int d> struct NoWarp {
  typedef PetscReal T;
  typedef Vector<T,d> TV;

  NoWarp(RawArray<const T> params) {
    GEODE_ASSERT(!params.size());
  }

  // Initial guess in world space at parameter X
  TV map(const TV& X) const {
    return X;
  }

  // Deformation gradient at zero extra displacement at parameter X
  T F(const TV& X) const {
    return 1;
  }
};

struct BendWarp {
  typedef PetscReal T;
  typedef Vector<T,3> TV;

  const T length;
  const T angle;
  const T angle_over_length;
  const T radius;

  BendWarp(RawArray<const T> params)
    : length((GEODE_ASSERT(params.size()==2),params[0]))
    , angle(params[1])
    , angle_over_length(angle/length)
    , radius(length/angle) {}

  TV map(const TV& X) const {
    const T theta = angle_over_length*X.z; // One length is the given rotation angle
    const T r = X.x+radius;
    return TV(r*cos(theta),
              X.y,
              r*sin(theta));
  }

  Matrix<T,3> F(const TV& X) const {
    const T theta = angle_over_length*X.z; // One length is the given rotation angle
    const T sr = angle_over_length*(X.x+radius);
    const T c = cos(theta), s = sin(theta);
    return Matrix<T,3>(c,0,s,0,1,0,-sr*s,0,sr*c);
  }
};

}
