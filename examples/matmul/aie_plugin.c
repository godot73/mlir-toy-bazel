#include "iree/hal/local/executable_plugin.h"
#include "matmul.h"

static int aie_matmul_f32_workgroup(void* params_ptr, void* context,
				void* reserved) {
  typedef struct {
    const float* restrict binding0;
    size_t binding0_offset;
    const float* restrict binding1;
    size_t binding1_offset;
    float* restrict binding2;
    size_t binding2_offset;
    size_t size_0;
    size_t size_1;
    size_t tid;
    uint32_t processor_id;
    const uint64_t* restrict processor_data;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  // The operation `iree_codegen.ukernel.generic` always operates
  // on a slice of the inputs to produce a slice of the output,
  // so the loop here just needs to iterate from `0` to `size`,
  // where `size` is the size of the slice to be executed by this call.
  for (size_t i = 0; i < params->size_0; ++i) {
    // The operation `iree_codegen.ukernel.generic` takes a slice of
    // the inputs and outputs as operands. So the `pointer` and `offset`
    // passed into this function represent the starting location of
    // where to read the data from for this invocation of the function.
    params->binding2[params->binding2_offset + i] =
        params->binding0[params->binding0_offset + i] *
        params->binding1[params->binding2_offset + i];
  }
  return 0;
}

// Called once for each plugin load and paired with a future call to unload.
// We don't do anything special here as this plugin is meant to represent a
// pure/stateless kernel library. Even in standalone mode we could allocate
// using environment->host_allocator, set an out_self pointer, and parse
// parameters.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t aie_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  *out_self = NULL;  // no state in this plugin
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
// In this sample it's a no-op as we don't have state.
static void aie_plugin_unload(void* self) {}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t aie_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name,
                                          "aie_matmul_f32") == 0) {
      params->out_fn_ptrs[i] = simple_mul_workgroup;
      params->out_fn_contexts[i] = NULL;  // no context used, could be self
    } else {
      if (is_optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }
  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

// Exported on the shared library and used by the runtime to query the plugin
// interface. When statically linking the plugin this is just a function that
// can be called and can have any name to allow for multiple plugins. When
// dynamically linking the exported symbol must be exactly this with no C++
// name mangling.
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      // Declares what library version is present: newer runtimes may support
      // loading older plugins but newer plugins cannot load on older runtimes.
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      // Name and description are used for tracing/logging/diagnostics.
      .name = "sample_aie_matmul",
      .description =
          "standalone AIE matmul plugin sample "
          "(aie_plugin.c)",
      // Standalone plugins must declare that they are standalone so that the
      // runtime can verify support.
      .features = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
      // Standalone plugins don't support sanitizers.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = standalone_plugin_load,
      .unload = standalone_plugin_unload,
      .resolve = standalone_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
