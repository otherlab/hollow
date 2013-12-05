// Lame constants
#pragma once

namespace hollow {

static inline double lame_mu(const double youngs_modulus, const double poissons_ratio) {
  return youngs_modulus/(2*(1+poissons_ratio));
}

static inline double lame_lambda(const double youngs_modulus, const double poissons_ratio) {
  return youngs_modulus*poissons_ratio/((1+poissons_ratio)*(1-2*poissons_ratio));
}

}
