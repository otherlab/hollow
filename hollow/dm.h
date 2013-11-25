// Wrapper around a DM
#pragma once

#include <hollow/vec.h>
#include <hollow/mat.h>
#include <hollow/model.h>
#include <geode/mesh/forward.h>
#include <geode/mesh/ids.h>
#include <geode/python/Ptr.h>
#include <petscdmplex.h>
namespace hollow {

struct DM : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;

  ::DM dm;

protected:
  DM(const ::DM dm); // Steals ownership
public:
  ~DM();

  MPI_Comm comm() const;
  const char* name() const;

  Ref<Vec> create_local_vector() const;
  Ref<Vec> create_global_vector() const;
  Ref<Mat> create_matrix() const;
};

struct DMPlex : public DM {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef DM Base;
  typedef PetscReal T;

protected:
  Ptr<const Model> model;

  DMPlex(const ::DM dm); // Steals ownership
public:
  ~DMPlex();

  Ref<DMPlex> clone() const;

  int dim() const;

  // Numbers of elements at each depth
  Array<const int> counts() const;

  // Refine using a volume constraint
  void volume_refine(const T max_volume);

  // Refine uniformly
  void uniform_refine(const int levels=1);
 
  // Distribute over all processors in the DM's (existing) communicator
  void distribute(const string& partitioner="chaco");

  // Mark boundary simplicies with the given label.
  // Returns the numbers of boundary faces and all boundary elements
  Vector<int,2> mark_boundary(const string& label);

  // Create and set the default section
  //   dim - spatial dimension
  //   fes - list of FE objects
  //   boundary_label - label of boundary conditions, if any
  //   boundary_fields - list of FE indices with boundary conditions (all assumed to be on the same points)
  void create_default_section(const vector<string>& names, const vector<Ref<const FE>>& fes,
                              const string& boundary_label, RawArray<const int> boundary_fields);

  // Set function and Jacobian evaluation based on a finite element model
  void set_model(const Model& model);

  // Compare a vec against an analytic function.  Requires a model.
  T L2_error_vs_exact(const Vec& v) const;
};

// Create an unrefined unit box DM
HOLLOW_EXPORT Ref<DMPlex> dmplex_unit_box(const MPI_Comm comm, const int dim);

// Convert a 2D or 3D shell mesh to a DM.
// Returns the DMPlex and the ordered list of representative halfedges (since corner meshes have no EdgeIds).
HOLLOW_EXPORT Tuple<Ref<DMPlex>,Array<const HalfedgeId>> dmplex_mesh(const MPI_Comm comm, const TriangleTopology& mesh, Array<const real,2> X);

}
