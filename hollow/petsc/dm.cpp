// Wrapper around a DM

#include <hollow/petsc/dm.h>
#include <hollow/petsc/mpi.h>
#include <geode/mesh/TriangleTopology.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/utility/Log.h>
#include <petscsnes.h>
namespace hollow {

typedef PetscReal T;
GEODE_DEFINE_TYPE(DM)
GEODE_DEFINE_TYPE(DMPlex)
using Log::cout;
using std::endl;

DM::DM(const ::DM dm)
  : dm(dm) {}

DM::~DM() {
  CHECK(DMDestroy(&dm));
}

DMPlex::DMPlex(const ::DM dm)
  : DM(dm) {
  DMType type;
  CHECK(DMGetType(dm,&type));
  GEODE_ASSERT(string(type)==DMPLEX);
}

DMPlex::~DMPlex() {}

int DMPlex::dim() const {
  int dim;
  CHECK(DMPlexGetDimension(dm,&dim));
  return dim;
}

Array<const int> DMPlex::counts() const {
  const int dim = this->dim();
  Array<int> counts(dim+1);
  for (const int d : range(dim+1)) {
    int lo, hi;
    CHECK(DMPlexGetDepthStratum(dm,d,&lo,&hi));
    counts[d] = hi-lo;
  }
  return counts;
}

Ref<DMPlex> DMPlex::clone() const {
  ::DM copy;
  CHECK(DMClone(dm,&copy));
  // Share coordinates and coordinate section
  PetscSection s;
  CHECK(DMGetCoordinateSection(dm,&s));
  CHECK(DMSetCoordinateSection(copy,s));
  ::Vec c;
  CHECK(DMGetCoordinatesLocal(dm,&c));
  CHECK(DMSetCoordinatesLocal(copy,c));
  return new_<DMPlex>(copy);
}

MPI_Comm DM::comm() const {
  return PetscObjectComm((PetscObject)dm);
}

const char* DM::name() const {
  const char* name;
  CHECK(PetscObjectGetName((PetscObject)dm,&name));
  return name;
}

Ref<Vec> DM::create_local_vector() const {
  ::Vec v;
  CHECK(DMCreateLocalVector(dm,&v));
  return new_<Vec>(v);
}

Ref<Vec> DM::create_global_vector() const {
  ::Vec v;
  CHECK(DMCreateGlobalVector(dm,&v));
  return new_<Vec>(v);
}

Ref<Mat> DM::create_matrix() const {
  ::Mat m;
  CHECK(DMCreateMatrix(dm,&m));
  return new_<Mat>(m);
}

static void set_dm(::DM& dst, ::DM src) {
  if (src) {
    const char* name;
    CHECK(PetscObjectGetName((PetscObject)dst,&name));
    CHECK(PetscObjectSetName((PetscObject)src,name));
    CHECK(DMDestroy(&dst));
    dst = src;
  }
}

void DMPlex::volume_refine(const T max_volume) {
  ::DM fine = 0;
  CHECK(DMPlexSetRefinementUniform(dm,PETSC_FALSE));
  CHECK(DMPlexSetRefinementLimit(dm,max_volume));
  CHECK(DMRefine(dm,comm(),&fine));
  set_dm(dm,fine);
}

void DMPlex::uniform_refine(const int levels) {
  CHECK(DMPlexSetRefinementUniform(dm,PETSC_TRUE));
  for (int i=0;i<levels;i++) {
    ::DM fine = 0;
    CHECK(DMRefine(dm,comm(),&fine));
    set_dm(dm,fine);
  }
}

void DMPlex::distribute(const string& partitioner) {
  ::DM dist;
  CHECK(DMPlexDistribute(dm,partitioner.c_str(),0,0,&dist));
  set_dm(dm,dist);
}

static DMLabel create_label(::DM dm, const string& label) {
  PetscBool has;
  CHECK(DMPlexHasLabel(dm,label.c_str(),&has));
  if (has)
    throw ValueError(format("DMPlex: label '%s' already exists",label));
  CHECK(DMPlexCreateLabel(dm,label.c_str()));
  DMLabel L;
  CHECK(DMPlexGetLabel(dm,label.c_str(),&L));
  return L;
}
static Vector<int,2> complete_label(::DM dm, DMLabel L) {
  Vector<int,2> counts;
  CHECK(DMLabelGetStratumSize(L,1,&counts.x));
  CHECK(DMPlexLabelComplete(dm,L));
  CHECK(DMLabelGetStratumSize(L,1,&counts.y));
  return counts;
}

Vector<int,2> DMPlex::mark_boundary(const string& label) {
  const auto L = create_label(dm,label);
  CHECK(DMPlexMarkBoundaryFaces(dm,L));
  return complete_label(dm,L);
}

Vector<int,2> DMPlex::mark(const string& label, RawArray<const int> indices) {
  const auto L = create_label(dm,label);
  for (const auto i : indices)
    CHECK(DMLabelSetValue(L,i,1));
  return complete_label(dm,L);
}

void DMPlex::create_default_section(const vector<string>& names, const vector<Ref<const FE>>& fes,
                                    const string& boundary_label, RawArray<const int> boundary_fields) {
  // Check consistency
  GEODE_ASSERT(!names.size() || names.size()==fes.size());
  const int dim = this->dim();
  for (const auto& fe : fes) {
    const int fe_dim = fe->spatial_dimension();
    GEODE_ASSERT(dim==fe_dim,format("Dimension mismatch: DM has %d, FE has %d",dim,fe_dim));
  }

  // Extract FE component counts
  Array<int> components(int(fes.size()));
  Array<int,2> dofs(int(fes.size()),dim+1);
  for (const int i : range(components.size())) {
    components[i] = fes[i]->components();
    dofs[i] = fes[i]->dofs();
  }

  // Grab boundary faces
  Array<IS> boundaries(boundary_fields.size());
  if (boundaries.size()) {
    DMLabel L;
    CHECK(DMPlexGetLabel(dm,boundary_label.c_str(),&L));
    if (!L)
      throw RuntimeError(format("DM::create_default_section: invalid boundary label '%s'",boundary_label));
    CHECK(DMLabelGetStratumIS(L,1,&boundaries[0]));
    if (!boundaries[0]) // Apparently a mesh without boundary: skip the boundary condition
      boundaries.clear();
    else
      for (const int i : range(1,boundaries.size()))
        boundaries[i] = boundaries[0];
  }

  // Create section
  PetscSection section;
  CHECK(DMPlexCreateSection(dm,dim,int(fes.size()),components.data(),dofs.data(),
    boundaries.size(),boundary_fields.data(),boundaries.data(),
    0,&section));
  if (names.size())
    for (const int i : range(int(names.size())))
      if (names[i].size())
        CHECK(PetscSectionSetFieldName(section,i,names[i].c_str()));
  CHECK(DMSetDefaultSection(dm,section));

  // Nearly done
  CHECK(PetscSectionDestroy(&section));
  if (boundaries.size())
    CHECK(ISDestroy(&boundaries[0]));
}

#if 0
void DMPlex::set_model(const Model& model, const bool use_energy) {
  this->model = ref(model);
  CHECK(DMSNESSetFunctionLocal(dm,DMPlexComputeResidualFEM,(void*)&model.fem));
  CHECK(DMSNESSetJacobianLocal(dm,DMPlexComputeJacobianFEM,(void*)&model.fem));
  if (use_energy)
    CHECK(DMSNESSetObjective(dm,[](SNES snes, ::Vec X, T* energy, void* ctx) {
      const DMPlex& self = *(const DMPlex*)ctx;
      const Model& model = *self.model;
      PetscQuadrature quad, quad_bd;
      if (model.e) {
        GEODE_ASSERT(model.fe.size()==1,"Energy computation is only implemented for a single finite element field");
        CHECK(PetscFEGetQuadrature(model.fep[0],&quad));
      }
      if (model.eb) {
        GEODE_ASSERT(model.fe_bd.size()==1,"Energy computation is only implemented for a single boundary finite element field");
        CHECK(PetscFEGetQuadrature(model.fep_bd[0],&quad_bd));
      }
      CHECK(DMPlexComputeScalarsFEM(self.dm,1,quad,model.e,quad_bd,model.eb,X,energy,&const_cast_(model.fem)));
      if (!isfinite(*energy)) // Replace inf with large numbers to avoid problems with Tao
        *energy = 1e10;
      return PetscErrorCode(0);
    },this));
}

void DMPlex::project(const vector<Ref<const Analytic>> fs, InsertMode mode, Vec& x) const {
  GEODE_ASSERT(model,"Can't project without a model");
  GEODE_ASSERT(model->fe.size()==fs.size());
  Array<void(*)(const T*,S*,void*)> fps(int(fs.size()),uninit);
  Array<void*> ctxs(int(fs.size()),uninit);
  for (const int i : range(int(fs.size()))) {
    GEODE_ASSERT(fs[i]->dim==model->dim);
    GEODE_ASSERT(fs[i]->count==model->fe[i]->components());
    fps[i] = fs[i]->f;
    ctxs[i] = fs[i]->ctx;
  }
  CHECK(DMPlexProjectFunction(dm,model->fep.data(),fps.data(),ctxs.data(),mode,x.v));
}

T DMPlex::L2_error_vs_exact(const Vec& v) const {
  GEODE_ASSERT(model);
  for (auto& exact : model->exact)
    GEODE_ASSERT(exact);
  T error;
  CHECK(DMPlexComputeL2Diff(dm,model->fep.data(),model->exact.data(),model->exact_contexts.data(),v.v,&error));
  return error;
}

void DMPlex::write_vtk(const string& filename, const Vec& v) const {
  GEODE_ASSERT(model,"Need a model in order to enforce boundary conditions");

  // Make a VTK viewer
  PetscViewer viewer;
  CHECK(PetscViewerCreate(comm(),&viewer));
  CHECK(PetscViewerSetType(viewer,PETSCVIEWERVTK));
  CHECK(PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_VTK));
  CHECK(PetscViewerFileSetName(viewer,filename.c_str()));

  // Grab local vector and update name
  ::Vec local;
  const char *name;
  CHECK(DMGetLocalVector(dm,&local));
  CHECK(PetscObjectGetName((PetscObject)v.v,&name));
  CHECK(PetscObjectSetName((PetscObject)local,name));

  // Apply boundary conditions
  CHECK(DMGlobalToLocalBegin(dm,v.v,INSERT_VALUES,local));
  CHECK(DMGlobalToLocalEnd(dm,v.v,INSERT_VALUES,local));
  CHECK(DMPlexProjectFunctionLocal(dm,model->fep.data(),model->fem.bcFuncs,model->fem.bcCtxs,INSERT_BC_VALUES,local));

  // Write everything to disk
  CHECK(VecView(local,viewer));

  // Clean up
  CHECK(DMRestoreLocalVector(dm,&local));
  CHECK(PetscViewerDestroy(&viewer));
}
#endif

Ref<DMPlex> dmplex_unit_box(const MPI_Comm comm, const int dim) {
  ::DM dm;
  CHECK(DMPlexCreateBoxMesh(comm,dim,PETSC_FALSE,&dm));
  return new_<DMPlex>(dm);
}

static inline HalfedgeId canon(const TriangleTopology& mesh, const HalfedgeId e) {
  const auto v = mesh.vertices(e);
  const auto r = mesh.reverse(e);
  return mesh.is_boundary(e) ? r
       : mesh.is_boundary(r) ? e
                   : v.x<v.y ? e : r;
}

Tuple<Ref<DMPlex>,Array<const HalfedgeId>> dmplex_mesh(const MPI_Comm comm, const TriangleTopology& mesh, Array<const real,2> X) {
  GEODE_ASSERT(mesh.is_garbage_collected(),"mesh must be garbage collected to ensure contiguous indices");
  const int nf = mesh.n_faces(),
            nv = mesh.n_vertices(),
            ne = mesh.n_edges();

  // Serial only for now
  GEODE_ASSERT(comm_size(comm)==1);

  // Create dm
  ::DM dm;
  CHECK(DMPlexCreate(comm,&dm));
  CHECK(DMPlexSetDimension(dm,2));

  // Allocate cones.  We index faces first, then vertices, then edges, as per petsc connection.
  CHECK(DMPlexSetChart(dm,0,nf+nv+ne));
  for (const int f : range(nf))
    CHECK(DMPlexSetConeSize(dm,f,3));
  for (const int e : nf+nv+range(ne))
    CHECK(DMPlexSetConeSize(dm,e,2));
  CHECK(DMSetUp(dm));

  // Set cones.  Since corner meshes lack edge ids, we create some first
  Hashtable<HalfedgeId,int> edge_id;
  Array<HalfedgeId> edges;
  for (const auto e : mesh.halfedges()) {
    const auto c = canon(mesh,e);
    if (e==c) {
      const int i = edges.append(e);
      edge_id.set(e,i);
      const auto v = mesh.vertices(e);
      CHECK(DMPlexSetCone(dm,nf+nv+i,(nf+vec(v.x.id,v.y.id)).data()));
    }
  }
  for (const auto f : mesh.faces()) {
    const auto e = mesh.halfedges(f);
    const auto cx = canon(mesh,e.x),
               cy = canon(mesh,e.y),
               cz = canon(mesh,e.z);
    CHECK(DMPlexSetCone(dm,f.id,(nf+nv+vec(edge_id.get(cx),
                                           edge_id.get(cy),
                                           edge_id.get(cz))).data()));
    CHECK(DMPlexSetConeOrientation(dm,f.id,vec(e.x==cx?0:-2,
                                               e.y==cy?0:-2,
                                               e.z==cz?0:-2).data()));
  }
  CHECK(DMPlexSymmetrize(dm));
  CHECK(DMPlexStratify(dm));

  // Set coordinates
  PetscSection Xs;
  CHECK(DMGetCoordinateSection(dm,&Xs));
  CHECK(PetscSectionSetChart(Xs,nf,nf+nv));
  for (const int v : nf+range(nv))
    CHECK(PetscSectionSetDof(Xs,v,X.n));
  CHECK(PetscSectionSetUp(Xs));
  int size;
  CHECK(PetscSectionGetStorageSize(Xs,&size));
  GEODE_ASSERT(size==X.flat.size());
  ::Vec Xv;
  CHECK(VecCreate(comm,&Xv));
  CHECK(VecSetSizes(Xv,size,PETSC_DETERMINE));
  CHECK(VecSetType(Xv,VECSTANDARD));
  T* Xp;
  CHECK(VecGetArray(Xv,&Xp));
  memcpy(Xp,X.data(),sizeof(T)*X.flat.size());
  CHECK(VecRestoreArray(Xv,&Xp));
  CHECK(DMSetCoordinatesLocal(dm,Xv));
  CHECK(VecDestroy(&Xv));

  // Check counts
  const auto result = new_<DMPlex>(dm);
  GEODE_ASSERT(asarray(vec(mesh.n_vertices(),mesh.n_edges(),mesh.n_faces()))==result->counts());

  // All done!
  return tuple(result,edges.const_());
}

}
using namespace hollow;

void wrap_dm() {
  {
    typedef hollow::DM Self;
    Class<Self>("DM")
      .GEODE_GET(comm)
      .GEODE_GET(name)
      .GEODE_METHOD(create_local_vector)
      .GEODE_METHOD(create_global_vector)
      .GEODE_METHOD(create_matrix)
      ;
  } {
    typedef hollow::DMPlex Self;
    Class<Self>("DMPlex")
      .GEODE_METHOD(clone)
      .GEODE_GET(dim)
      .GEODE_GET(counts)
      .GEODE_METHOD(volume_refine)
      .GEODE_METHOD(uniform_refine)
      .GEODE_METHOD(distribute)
      .GEODE_METHOD(mark_boundary)
      .GEODE_METHOD(mark)
      .GEODE_METHOD(create_default_section)
#if 0
      .GEODE_METHOD(set_model)
      .GEODE_METHOD(project)
      .GEODE_METHOD(L2_error_vs_exact)
      .GEODE_METHOD(write_vtk)
#endif
      ;
  }

  GEODE_FUNCTION(dmplex_unit_box)
  GEODE_FUNCTION(dmplex_mesh)
}
