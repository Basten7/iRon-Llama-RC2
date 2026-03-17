#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include <vector>
#include <string>
#include <cstdio>
#include <mutex>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
// globals


static bool ggml_metal_env_offload_debug_(void) {
    const char * v = getenv("GGML_METAL_OFFLOAD_DEBUG");
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}
static bool ggml_metal_env_trace_ops_(void) {
    const char * v = getenv("GGML_METAL_TRACE_OPS");
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

static const char * ggml_metal_op_name_trace_(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:        return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID:     return "MUL_MAT_ID";
        case GGML_OP_ADD_ID:         return "ADD_ID";
        case GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        default:                     return "OTHER";
    }
}

static void ggml_metal_trace_op_decision_(
        const char * where,
        ggml_backend_dev_t dev,
        const ggml_tensor * op,
        int64_t bs,
        int decision,
        const char * reason) {
    if (!ggml_metal_env_trace_ops_()) {
        return;
    }

    const char * dev_name = ggml_backend_dev_description(dev);
    const char * op_name  = op ? ggml_metal_op_name_trace_(op->op) : "(null)";
    const char * tname    = (op && op->name[0]) ? op->name : "(unnamed)";

    fprintf(stderr,
        "ggml-metal trace: where=%s dev='%s' op=%s tensor='%s' bs=%lld ne=[%lld,%lld,%lld,%lld] decision=%d reason=%s\n",
        where ? where : "(null)",
        dev_name ? dev_name : "(unknown)",
        op_name,
        tname,
        (long long) bs,
        (long long) (op ? op->ne[0] : -1),
        (long long) (op ? op->ne[1] : -1),
        (long long) (op ? op->ne[2] : -1),
        (long long) (op ? op->ne[3] : -1),
        decision,
        reason ? reason : "(none)");
}
static const char * ggml_metal_op_name_(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:        return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID:     return "MUL_MAT_ID";
        case GGML_OP_ADD_ID:         return "ADD_ID";
        case GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        default:                     return "OTHER";
    }
}

static void ggml_metal_log_offload_decision_(
        ggml_backend_dev_t dev,
        const ggml_tensor * op,
        int64_t bs,
        bool decision,
                                             const char * reason) {
    if (!ggml_metal_env_offload_debug_()) {
        return;
    }
    
    const char * tensor_name = op && op->name[0] ? op->name : "(unnamed)";
    const char * op_name     = op ? ggml_metal_op_name_(op->op) : "(null)";
    const char * dev_name    = ggml_backend_dev_description(dev);
    
    const int64_t ne0 = op ? op->ne[0] : -1;
    const int64_t ne1 = op ? op->ne[1] : -1;
    const int64_t ne2 = op ? op->ne[2] : -1;
    const int64_t ne3 = op ? op->ne[3] : -1;
    
    //    GGML_LOG_INFO(
    //        "ggml-metal offload: dev='%s' op=%s tensor='%s' bs=%lld ne=[%lld,%lld,%lld,%lld] -> %s (%s)\n",
    //        dev_name ? dev_name : "(unknown)",
    //        op_name,
    //        tensor_name,
    //        (long long) bs,
    //        (long long) ne0,
    //        (long long) ne1,
    //        (long long) ne2,
    //        (long long) ne3,
    //        decision ? "METAL" : "CPU",
    //        reason ? reason : "no-reason");
    
    fprintf(stderr,
            "ggml-metal offload: dev='%s' op=%s tensor='%s' bs=%lld ne=[%lld,%lld,%lld,%lld] -> %s (%s)\n",
            dev_name ? dev_name : "(unknown)",
            op_name,
            tensor_name,
            (long long) bs,
            (long long) ne0,
            (long long) ne1,
            (long long) ne2,
            (long long) ne3,
            decision ? "METAL" : "CPU",
            reason ? reason : "no-reason");
}
static bool ggml_metal_op_is_decode_like_(const ggml_tensor * op) {
    if (op == NULL) {
        return false;
    }

    if (op->op == GGML_OP_MUL_MAT) {
        return op->ne[1] <= 1;
    }
    if (op->op == GGML_OP_MUL_MAT_ID) {
        return op->ne[2] <= 1;
    }
    return false;
}

static int ggml_metal_env_mgpu_small_mat_offload_max_bs_(void) {
    const char * v = getenv("GGML_METAL_MGPU_SMALL_MAT_OFFLOAD_MAX_BS");
    if (v == NULL || v[0] == '\0') {
        return -1; // disabled by default
    }

    const int x = atoi(v);
    if (x < 0) return -1;
    return x;
}

static int ggml_metal_env_mgpu_decode_offload_min_bs_(void) {
    const char * v = getenv("GGML_METAL_MGPU_DECODE_OFFLOAD_MIN_BS");
    if (v == NULL || v[0] == '\0') {
        return -1; // use device default
    }

    const int x = atoi(v);
    if (x < 0) return 0;
    return x;
}

static bool ggml_metal_env_is_mgpu_enabled_(void) {
    const char * v = getenv("GGML_METAL_DEVICE_INDEX");
    return v != NULL && strchr(v, ',') != NULL;
}

static int ggml_metal_env_mgpu_mul_mat_decode_min_bs_(void) {
    const char * v = getenv("GGML_METAL_MGPU_MUL_MAT_DECODE_MIN_BS");
    if (v == NULL || v[0] == '\0') {
        return 16;
    }
    
    const int x = atoi(v);
    
    // Clamp to a sane range:
    // 0  => fully reopen MUL_MAT decode offload
    // 1+ => keep a minimum batch-size guard
    if (x < 0) return 0;
    return x;
}
// initialized in ggml_backend_metal_reg
static ggml_backend_reg g_ggml_metal_reg;

// MGPU: expose multiple devices (Metal0..MetalN-1) from a single registry.
static std::vector<ggml_backend_device> g_ggml_metal_devices;

// Stable "device0" for legacy buffer types and safe fallbacks
static ggml_backend_device g_ggml_metal_device;
static std::once_flag g_ggml_metal_reg_once;

// Device indices requested by env (defaults to single device index).
static std::vector<int>                g_ggml_metal_dev_indices;

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return false;
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer     = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base        = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor      = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor      = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear           = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset           = */ NULL,
};
// Mapped buffers use the shared buffer interface in this branch.
static ggml_backend_buffer_i ggml_backend_metal_buffer_mapped_i = ggml_backend_metal_buffer_shared_i;

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return false;
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer     = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base        = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor      = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor      = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear           = */ ggml_backend_metal_buffer_private_clear,
    /* .reset           = */ NULL,
};

//
// buffer types
//

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft == NULL || buft->device == NULL || buft->device->context == NULL) {
        return SIZE_MAX;
    }
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) buft->device->context;
    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_shared_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_shared_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_shared_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_shared_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_metal;
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal_Private";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft == NULL || buft->device == NULL || buft->device->context == NULL) { return SIZE_MAX; }
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_private_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_private_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_private_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_private_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_metal;
}

// MGPU: buffer types must be per-device because ggml_backend_buffer_type_t contains a back-pointer to the device.
struct ggml_metal_buft_set {
    ggml_backend_buffer_type shared;
    ggml_backend_buffer_type priv;
    ggml_backend_buffer_type mapped;
};

static std::mutex g_ggml_metal_buft_mu;
static std::unordered_map<ggml_backend_dev_t, ggml_metal_buft_set> g_ggml_metal_bufts;

// Forward decl: used by *_mapped_for_dev_ before definition below.
static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(void);

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped_for_dev_(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared_for_dev_(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lock(g_ggml_metal_buft_mu);
    auto & set = g_ggml_metal_bufts[dev];
    // Defensive: ensure a clean zero state if the struct ever changes ABI
    // (keeps .device checks reliable).
    if (set.shared.device == NULL && set.priv.device == NULL && set.mapped.device == NULL) {
        // no-op in practice (operator[] value-initializes), but keeps intent explicit
    }
    if (set.shared.device == NULL) {
        set.shared = {
            /* .iface   = */ {
                /* .get_name         = */ ggml_backend_metal_buffer_type_shared_get_name,
                /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                /* .get_alignment    = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                /* .get_max_size     = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                /* .is_host          = */ ggml_backend_metal_buffer_type_shared_is_host,
            },
            /* .device  = */ dev,
            /* .context = */ NULL,
        };
    }
    return &set.shared;
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private_for_dev_(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lock(g_ggml_metal_buft_mu);
    auto & set = g_ggml_metal_bufts[dev];
    if (set.priv.device == NULL) {
        set.priv = {
            /* .iface   = */ {
                /* .get_name         = */ ggml_backend_metal_buffer_type_private_get_name,
                /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                /* .get_alignment    = */ ggml_backend_metal_buffer_type_private_get_alignment,
                /* .get_max_size     = */ ggml_backend_metal_buffer_type_private_get_max_size,
                /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                /* .is_host          = */ ggml_backend_metal_buffer_type_private_is_host,
            },
            /* .device  = */ dev,
            /* .context = */ NULL,
        };
    }
    return &set.priv;
}
// Mapped buft is a per-device view of the global mapped interface, with .device fixed to `dev`.
static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped_for_dev_(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lock(g_ggml_metal_buft_mu);
    auto & set = g_ggml_metal_bufts[dev];
    if (set.mapped.device == NULL) {
        // Copy the global mapped buft (iface is identical), then pin the device pointer.
        set.mapped = *ggml_backend_metal_buffer_type_mapped();
        set.mapped.device  = dev;
        set.mapped.context = NULL;
    } else {
        // Keep device coherent if dev address ever changes (shouldn't, but safe).
        set.mapped.device = dev;
    }
    return &set.mapped;
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft == NULL || buft->device == NULL || buft->device->context == NULL) { return SIZE_MAX; }
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(void) {
    // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
    //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
    static ggml_backend_buffer_type ggml_backend_buffer_type_mapped_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_mapped_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_mapped_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_mapped_metal;
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    return "Metal";

    GGML_UNUSED(backend);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!backend_src || !backend_dst || !src || !dst) {
        return false;
    }
    
    const bool src_is_metal = ggml_backend_is_metal(backend_src);
    const bool dst_is_metal = ggml_backend_is_metal(backend_dst);
    
    const size_t nbytes = ggml_nbytes(src);
    if (nbytes == 0) {
        return true;
    }
    GGML_ASSERT(ggml_nbytes(dst) == nbytes);
    
    // Only handle cases where at least one side is Metal.
    if (!src_is_metal && !dst_is_metal) {
        return false;
    }
    
    std::vector<uint8_t> tmp;
    tmp.resize(nbytes);
    
    // --- src -> host ---
    if (src_is_metal) {
        ggml_metal_t ctx_src = (ggml_metal_t) backend_src->context;
        if (ctx_src == NULL) {
            return false;
        }
        ggml_metal_get_tensor_async(ctx_src, src, tmp.data(), 0, nbytes);
        ggml_metal_synchronize(ctx_src);
    } else {
        // src is non-Metal (CPU): use generic backend getter
        ggml_backend_tensor_get(src, tmp.data(), 0, nbytes);
    }
    
    // --- host -> dst ---
    if (dst_is_metal) {
        ggml_metal_t ctx_dst = (ggml_metal_t) backend_dst->context;
        if (ctx_dst == NULL) {
            return false;
        }
        ggml_metal_set_tensor_async(ctx_dst, dst, tmp.data(), 0, nbytes);
        ggml_metal_synchronize(ctx_dst);
    } else {
        // dst is non-Metal (CPU): use generic backend setter
        ggml_backend_tensor_set(dst, tmp.data(), 0, nbytes);
    }
    
    return true;
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);

}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,

    // the events API is needed only for multi-GPU setups, so likely no need to implement it for Metal
    // in any case, these docs seem relevant if we ever decide to implement it:
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer#Synchronizing-Passes-with-Events
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    if (reg == NULL) {
        GGML_LOG_ERROR("%s: error: Metal backend registry unavailable\n", __func__);
        return NULL;
    }
    
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (dev == NULL || dev->context == NULL) {
        GGML_LOG_ERROR("%s: error: Metal device unavailable\n", __func__);
        return NULL;
    }
             
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) dev->context;
    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

ggml_backend_t ggml_backend_metal_init_by_index(size_t reg_dev_index) {
ggml_backend_reg_t reg = ggml_backend_metal_reg();
if (reg == NULL) {
    GGML_LOG_ERROR("%s: error: Metal backend registry unavailable\n", __func__);
    return NULL;
}

const size_t n_dev = ggml_backend_reg_dev_count(reg);
if (n_dev == 0 || reg_dev_index >= n_dev) {
    GGML_LOG_ERROR("%s: error: invalid Metal registry device index %zu (device_count=%zu)\n",
                   __func__, reg_dev_index, n_dev);
    return NULL;
}

ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, reg_dev_index);
if (dev == NULL || dev->context == NULL) {
    GGML_LOG_ERROR("%s: error: Metal device unavailable (reg_dev_index=%zu)\n", __func__, reg_dev_index);
    return NULL;
}

ggml_metal_device_t ctx_dev = (ggml_metal_device_t) dev->context;
ggml_metal_t ctx = ggml_metal_init(ctx_dev);
if (ctx == NULL) {
    GGML_LOG_ERROR("%s: error: failed to allocate context (reg_dev_index=%zu)\n", __func__, reg_dev_index);
    return NULL;
}

ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));
*backend = {
    /* .guid      = */ ggml_backend_metal_guid(),
    /* .interface = */ ggml_backend_metal_i,
    /* .device    = */ dev,
    /* .context   = */ ctx,
};

ggml_backend_metal_set_n_cb(backend, 1);
return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    return "Metal";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->name;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    const bool is_mgpu = ggml_metal_env_is_mgpu_enabled_();
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        // In Metal MGPU, advertising async=true while events=false can leave the scheduler
        // in a fragile state during decode/TG. Keep async in single-GPU mode, but force
        // non-async semantics in MGPU until Metal events or equivalent synchronization exist.
        /* .async                 = */ !is_mgpu,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_metal_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;
    
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);
    
    return props_dev->use_shared_buffers
    ? ggml_backend_metal_buffer_type_shared_for_dev_(dev)
    : ggml_backend_metal_buffer_type_private_for_dev_(dev);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    // IMPORTANT: backend buffer context must be the Metal buffer object, not the raw host pointer.
    // The mapped buffer interface uses ggml_metal_buffer_* helpers which expect ggml_metal_buffer_t.
    // Important: buffer context must be the Metal buffer object
    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped_for_dev_(dev), ggml_backend_metal_buffer_mapped_i, res, size);
}

    static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
        ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

        if (op == NULL) {
            ggml_metal_trace_op_decision_("supports_op", dev, op, -1, 0, "op-null");
            return false;
        }

        const int64_t bs =
            op->op == GGML_OP_MUL_MAT    ? op->ne[1] :
            op->op == GGML_OP_MUL_MAT_ID ? op->ne[2] :
            ggml_nrows(op);
        
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_ADD_ID) {
            ggml_metal_trace_op_decision_("supports_op-enter", dev, op, bs, 1, "enter");
        }

        const bool ok = ggml_metal_device_supports_op(ctx_dev, op);

        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_ADD_ID) {
            ggml_metal_trace_op_decision_("supports_op-exit", dev, op, bs, ok ? 1 : 0,
                ok ? "device-supports-op-yes" : "device-supports-op-no");
        }

        return ok;
    }

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    const bool is_metal_buft =
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name;
    if (!is_metal_buft) {
        return false;
    }
    // In MGPU, Metal buffer types are per-device.
    // A Metal backend device must only accept Metal buffer types whose back-pointer
    // matches that exact device. Accepting a Metal buft from another Metal device
    // can later produce cross-device resource use and command-buffer/resource mismatches.
    if (ggml_metal_env_is_mgpu_enabled_()) {
        return buft != NULL && buft->device == dev;
    }
    // Single-GPU / legacy fallback:
    // keep previous permissive behavior for compatibility with legacy singleton bufts.
    return true;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const bool    is_mgpu            = ggml_metal_env_is_mgpu_enabled_();
    const int     mgpu_decode_min_bs = ggml_metal_env_mgpu_mul_mat_decode_min_bs_();
    const int64_t bs                 = get_op_batch_size(op);
    const bool    is_decode_like     = ggml_metal_op_is_decode_like_(op);
    const int64_t offload_min_bs     = ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
    int64_t       effective_offload_min_bs = offload_min_bs;
    const int     small_mat_offload_max_bs = ggml_metal_env_mgpu_small_mat_offload_max_bs_();
    bool          small_mat_override = false;    if (op == NULL) {
        ggml_metal_log_offload_decision_(dev, op, -1, false, "op-null");
        return false;
    }

    // Limit this hook to matmul-like ops only.
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        ggml_metal_log_offload_decision_(dev, op, bs, false, "unsupported-op-class");
        return false;
    }

    // Single-GPU: preserve existing behavior for both MUL_MAT and MUL_MAT_ID.
    if (!is_mgpu) {
        const bool ok = bs >= offload_min_bs;
        ggml_metal_log_offload_decision_(
            dev, op, bs, ok,
            ok ? "single-gpu-batch-ok" : "single-gpu-batch-too-small");
        return ok;
    }

    // MGPU policy:
    // - reopen PP-only offload for MUL_MAT_ID
    // - keep decode-like tiny-batch matmul/id on CPU for safety
    if (is_decode_like && bs <= mgpu_decode_min_bs) {
        ggml_metal_log_offload_decision_(dev, op, bs, false, "mgpu-decode-guard");
        return false;
    }
    // MGPU decode override:
    // allow small decode-like matmul paths to bypass the generic device threshold
    // when explicitly requested by env. This is meant for TG/MoE investigation.
    if (is_mgpu && is_decode_like &&
        (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID)) {
        const int env_decode_offload_min_bs = ggml_metal_env_mgpu_decode_offload_min_bs_();
        if (env_decode_offload_min_bs >= 0) {
            effective_offload_min_bs = env_decode_offload_min_bs;
        }
    }
    // MGPU small-mat override:
    // allow small TG/MoE matmul paths (e.g. bs=16) to bypass the generic device
    // threshold even when they are not classified as decode-like.
    if (is_mgpu &&
        (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) &&
        small_mat_offload_max_bs >= 0 &&
        bs <= small_mat_offload_max_bs) {
        const int env_decode_offload_min_bs = ggml_metal_env_mgpu_decode_offload_min_bs_();
        if (env_decode_offload_min_bs >= 0) {
            effective_offload_min_bs = env_decode_offload_min_bs;
            small_mat_override = true;
        }
    }
    
    if (ggml_metal_env_offload_debug_()) {
        fprintf(stderr,
            "ggml-metal offload-threshold: dev='%s' op=%s tensor='%s' bs=%lld offload_min_bs=%lld effective_offload_min_bs=%lld is_mgpu=%d is_decode_like=%d small_mat_offload_max_bs=%d small_mat_override=%d\n",
            ggml_backend_dev_description(dev),
            ggml_metal_op_name_(op->op),
            op->name[0] ? op->name : "(unnamed)",
            (long long) bs,
            (long long) offload_min_bs,
            (long long) effective_offload_min_bs,
            (int) is_mgpu,
            (int) is_decode_like,
            (int) small_mat_offload_max_bs,
            (int) small_mat_override);
    }
    if (bs < effective_offload_min_bs) {
        ggml_metal_log_offload_decision_(dev, op, bs, false, "device-offload-min-bs");
        return false;
    }

    ggml_metal_log_offload_decision_(
        dev, op, bs, true,
        op->op == GGML_OP_MUL_MAT_ID ? "mgpu-pp-reopen-mul_mat_id" : "mgpu-batch-ok");

    return true;
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return "Metal";

    GGML_UNUSED(reg);
}
static std::vector<int> ggml_metal_parse_device_list_from_env_(const char * env_name) {
    std::vector<int> out;

    const char * v = getenv(env_name);
    if (!v || !v[0]) {
        out.push_back(0);
        return out;
    }

    // Accept formats:
    //  - "2"
    //  - "0,1,2,3"
    //  - "0, 1, 2, 3"
    const char * p = v;
    while (*p) {
        while (*p && (std::isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;

        bool neg = false;
        if (*p == '-') { neg = true; p++; }

        int val = 0;
        bool any = false;
        while (*p && std::isdigit((unsigned char)*p)) {
            any = true;
            val = val * 10 + (*p - '0');
            p++;
        }
        if (any) {
            if (neg) val = -val;
            out.push_back(val);
        }

        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }

    if (out.empty()) out.push_back(0);
    return out;
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    // Be defensive: if the list is empty, still expose a single fallback device
    // (it will resolve to ggml_metal_device_get() / first index).
    return g_ggml_metal_devices.empty() ? 1 : g_ggml_metal_devices.size();

    GGML_UNUSED(reg);
}
static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    if (g_ggml_metal_devices.empty()) {
        // Fallback: expose device0 (or initialize it lazily)
        if (g_ggml_metal_device.context == NULL) {
            g_ggml_metal_device = {
                /* .iface   = */ ggml_backend_metal_device_i,
                /* .reg     = */ &g_ggml_metal_reg,
                /* .context = */ ggml_metal_device_get(),
            };
        }
        GGML_ASSERT(index == 0);
        return &g_ggml_metal_device;
    }
    
    GGML_ASSERT(index < g_ggml_metal_devices.size());
    return &g_ggml_metal_devices[index];

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .device_count     = */ ggml_backend_metal_reg_device_count,
    /* .device_get       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static std::vector<int> ggml_backend_metal_parse_device_indices_(void) {
    const char * s = getenv("GGML_METAL_DEVICE_INDEX");
    std::vector<int> out;
    if (!s || !s[0]) {
        out.push_back(0);
        return out;
    }

    std::unordered_set<int> seen;
    const char * p = s;
    while (*p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == ',' || *p == ';' || *p == ':')) ++p;
        if (!*p) break;
        char * end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        p = end;
        if (v < 0) continue;
        if (seen.insert((int)v).second) out.push_back((int)v);
    }
    if (out.empty()) out.push_back(0);
    return out;
}

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    std::call_once(g_ggml_metal_reg_once, []() {
        g_ggml_metal_reg = {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_metal_reg_i,
            /* .context     = */ NULL,
        };

        const std::vector<int> indices = ggml_backend_metal_parse_device_indices_();
        g_ggml_metal_devices.clear();
        g_ggml_metal_devices.reserve(indices.size());

        for (int idx : indices) {
            ggml_metal_device_t dev = ggml_metal_device_get_by_index(idx);
            if (!dev) {
                GGML_LOG_ERROR("ggml-metal: device index %d unavailable; skipping\n", idx);
                continue;
            }
            ggml_backend_device d = { ggml_backend_metal_device_i, &g_ggml_metal_reg, dev };
            g_ggml_metal_devices.push_back(d);
        }

        if (!g_ggml_metal_devices.empty()) {
            g_ggml_metal_device = g_ggml_metal_devices[0];
        } else {
            g_ggml_metal_device = { ggml_backend_metal_device_i, &g_ggml_metal_reg, ggml_metal_device_get() };
        }

        if (g_ggml_metal_device.context == NULL) {
            GGML_LOG_ERROR("ggml-metal: no Metal device available; registry will be disabled\n");
            return;
        }

        if (g_ggml_metal_devices.size() > 1) {
            GGML_LOG_INFO("ggml-metal: registered %zu Metal devices from GGML_METAL_DEVICE_INDEX='%s'\n",
                          g_ggml_metal_devices.size(),
                          getenv("GGML_METAL_DEVICE_INDEX") ? getenv("GGML_METAL_DEVICE_INDEX") : "");
        }
    });

    return (g_ggml_metal_device.context != NULL) ? &g_ggml_metal_reg : NULL;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
