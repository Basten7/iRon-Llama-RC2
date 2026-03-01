//ggml-metal-device.fixed.cpp le 8/02/2026 à 14-45

#include "ggml-metal-device.h"

#include "ggml-metal-impl.h"

#include "ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <cstdlib>



// -----------------------------------------------------------------------------
// Decode-only env helpers (must be visible before ggml_metal_library_get_pipeline_mul_mv)
// Prefer env_decode if set, else fall back to env_global.
static inline const char * ggml_metal_getenv_pref_(const char * env_decode, const char * env_global) {
    const char * v = getenv(env_decode);
    if (v && v[0]) {
        return v;
    }
    v = getenv(env_global);
    if (v && v[0]) {
        return v;
    }
    return NULL;
}

// NR0 is compile-time in Metal kernels: only a discrete set exists.
// Pick the greatest allowed NR0 <= max_nr0 (and >=2).
static inline int ggml_metal_nr0_allowed_leq_(int max_nr0) {
    if (max_nr0 >= 256) return 256;
    if (max_nr0 >= 128) return 128;
    if (max_nr0 >=  64) return  64;
    if (max_nr0 >=  32) return  32;
    if (max_nr0 >=  16) return  16;
    if (max_nr0 >=   8) return   8;
    if (max_nr0 >=   4) return   4;
    return 2;
}
// -----------------------------------------------------------------------------


struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(void) {
    static std::once_flag s_once;
    static ggml_metal_device_ptr s_dev;
    
    std::call_once(s_once, []() {
        s_dev.reset(ggml_metal_device_init());
        if (!s_dev) {
            GGML_LOG_ERROR("ggml-metal: ggml_metal_device_init() returned NULL\n");
        }
    });
    
    // Defensive: allow a best-effort retry if first init failed (rare, but prevents NULL being sticky forever)
    if (!s_dev) {
        ggml_metal_device_t tmp = ggml_metal_device_init();
        if (tmp) {
            s_dev.reset(tmp);
            GGML_LOG_ERROR("ggml-metal: device init succeeded on retry\n");
        }
    }
    
    return s_dev.get();
}

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    if (!ppls) {
        return;
    }

    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        case GGML_OP_CONCAT: op_str = "concat"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, sizeof(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, ggml_type tidx, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_set_rows_%s_%s", ggml_type_name(tdst), ggml_type_name(tidx));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    const int64_t n = ggml_nelements(op);

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_SCALE:      op_str = "scale";      break;
        case GGML_OP_FILL:       op_str = "fill";       break;
        case GGML_OP_CLAMP:      op_str = "clamp";      break;
        case GGML_OP_SQR:        op_str = "sqr";        break;
        case GGML_OP_SQRT:       op_str = "sqrt";       break;
        case GGML_OP_SIN:        op_str = "sin";        break;
        case GGML_OP_COS:        op_str = "cos";        break;
        case GGML_OP_LOG:        op_str = "log";        break;
        case GGML_OP_LEAKY_RELU: op_str = "leaky_relu"; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_str = "tanh";        break;
                case GGML_UNARY_OP_RELU:        op_str = "relu";        break;
                case GGML_UNARY_OP_SIGMOID:     op_str = "sigmoid";     break;
                case GGML_UNARY_OP_GELU:        op_str = "gelu";        break;
                case GGML_UNARY_OP_GELU_ERF:    op_str = "gelu_erf";    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_str = "gelu_quick";  break;
                case GGML_UNARY_OP_SILU:        op_str = "silu";        break;
                case GGML_UNARY_OP_ELU:         op_str = "elu";         break;
                case GGML_UNARY_OP_NEG:         op_str = "neg";         break;
                case GGML_UNARY_OP_ABS:         op_str = "abs";         break;
                case GGML_UNARY_OP_SGN:         op_str = "sgn";         break;
                case GGML_UNARY_OP_STEP:        op_str = "step";        break;
                case GGML_UNARY_OP_HARDSWISH:   op_str = "hardswish";   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_str = "hardsigmoid"; break;
                case GGML_UNARY_OP_EXP:         op_str = "exp";         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_str = "softplus";    break;
                case GGML_UNARY_OP_EXPM1:       op_str = "expm1";       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * suffix = "";
    if (n % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_%s_%s%s", op_str, ggml_type_name(op->src[0]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_SUM_ROWS:
            op_str = "sum_rows"; break;
        case GGML_OP_MEAN:
            op_str = "mean"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "tri";
    const int ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(ggml_metal_library_t lib, const ggml_tensor * op, int ssm_conv_bs) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d", base, ssm_conv_bs);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, const ggml_tensor * op)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    char base[256];
    char name[256];

    const int nsg = (ne00 + 31)/32;

    snprintf(base, 256, "kernel_ssm_scan_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = (32 + 2)*sizeof(float)*nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1, int nsg, int nxpsg, int r1ptg) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d", base, nsg, nxpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,   FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg, FC_MUL_MV + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;
    const bool bc_out = op->ne[0] % 64 != 0 || op->ne[1] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d", base, bc_inp, bc_out);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // when the output size is not multiple of 64x32, we need extra smem to prevent out-of-bounds writes
    res.smem = bc_out ? 8192 : 4096 + 2048;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    // Decode path identification must match the _id_ variant (mul_mv_id) to ensure that
    // GGML_METAL_DECODE_Q4K_NR0/NSG affect BOTH pipelines (id + non-id).
    //
    // In llama.cpp, many "matvec decode-like" ops have ne11 == 1 but may carry extra
    // batching/splitting in ne12/ne13, so restricting decode to (ne12==1 && ne13==1)
    // causes overrides to be silently ignored.
    // Decode heuristics for env overrides (NR0/NSG tuning):
    // - strict decode: classic mat-vec with no extra batching in higher dims
    // - decode-like: still a "mat-vec" (ne11==1) but can carry light batching/splitting in ne12/ne13
    //   We keep this bounded to avoid misclassifying prefill/small-batch paths.
    const bool is_decode_strict = (ne11 == 1 && ne12 == 1 && ne13 == 1);
    const int64_t aux_batch      = (int64_t) ne12 * (int64_t) ne13;
    const bool is_decode_like    = (ne11 == 1 && aux_batch <= 8);
    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
// Optional override: rows per threadgroup for F32/F16 mat-vec (non-short path).
// This maps to args.nr0 consumed by the Metal kernel dispatch switch.
// Usage:
//   export GGML_METAL_F32_NR0=8
//   export GGML_METAL_F16_NR0=8
                                if (strcmp(suffix, "_short") != 0) {
                                    const bool is_f32 = (tsrc0 == GGML_TYPE_F32);
                                    const bool is_f16 = (tsrc0 == GGML_TYPE_F16);
                                    const char * env = is_f32 ? getenv("GGML_METAL_F32_NR0")
                                                     : is_f16 ? getenv("GGML_METAL_F16_NR0")
                                                              : NULL;
                                    if (env) {
                                        const int v = atoi(env);
                                        // Supported NR0 values are those compiled into the Metal dispatch switch.
                                        if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32) {
                                            nr0  = v;
                                            smem = 32*sizeof(float)*nr0;
                                        }
                                        static bool s_logged_f32 = false;
                                        static bool s_logged_f16 = false;
                                        bool & s_logged = is_f32 ? s_logged_f32 : s_logged_f16;
                                        if (!s_logged) {
                                            s_logged = true;
                                            GGML_LOG_INFO("%s: %s NR0 override = %d\n",
                                                          __func__, is_f32 ? "F32" : "F16", (int) nr0);
                                        }
                                    }
                                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
                // Decode-preferred envs must affect BOTH Q4_K pipelines (mul_mv + mul_mv_id):
                // - decode: prefer GGML_METAL_DECODE_Q4K_* then fallback to GGML_METAL_Q4K_*
                // - non-decode: only GGML_METAL_Q4K_*
                const char * env_q4k_nsg =
                is_decode_like ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q4K_NSG", "GGML_METAL_Q4K_NSG")
                            : getenv("GGML_METAL_Q4K_NSG");
                const char * env_q4k_nr0 =
                is_decode_like ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q4K_NR0", "GGML_METAL_Q4K_NR0")
                            : getenv("GGML_METAL_Q4K_NR0");
                
                if (env_q4k_nsg && env_q4k_nsg[0]) {
                    const int v = atoi(env_q4k_nsg);
                    if (v >= 1 && v <= 8) {
                        nsg = v;
                    }
                }
                
                if (env_q4k_nr0 && env_q4k_nr0[0]) {
                    const int v_req = atoi(env_q4k_nr0);
                    // Only values with dedicated entry points should be selected.
                    // If the user asks something else, clamp down to the nearest allowed <= request.
                    const int v = ggml_metal_nr0_allowed_leq_(v_req);
                    if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128 || v == 256) {
                        nr0 = v;
                    }
                }
                
                // Safety clamp: avoid invalid (ne00 < nsg*nr0) combos that would select a kernel
                // specialization incompatible with the actual problem size (common with tiny heads).
                // Keep behavior consistent with Q6_K: clamp NR0 down to the nearest allowed <= (ne00/nsg).
                if (nsg < 1) nsg = 1;
                if (nr0 < 2) nr0 = 2;
                const int max_nr0 = (int) (ne00 / nsg);
                if (max_nr0 >= 2) {
                    const int nr0_clamped = ggml_metal_nr0_allowed_leq_(max_nr0);
                    if (nr0 > nr0_clamped) {
                        nr0 = nr0_clamped;
                    }
                } else {
                    // Degenerate: force a minimal safe config.
                    nsg = 1;
                    nr0 = 2;
                }
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
                // Optional override: simdgroups per threadgroup for Q6_K mat-vec.
                // Usage: export GGML_METAL_Q6K_NSG=4 (or GGML_METAL_DECODE_Q6K_NSG=4 for decode only)
                const char * env_q6k_nsg =
                +is_decode_like ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q6K_NSG", "GGML_METAL_Q6K_NSG")
                                   : getenv("GGML_METAL_Q6K_NSG");
                if (env_q6k_nsg) {
                    const int v = atoi(env_q6k_nsg);
                    if (v >= 1 && v <= 8) {
                        nsg = v;
                    }
                }
                
                // Optional override: rows per threadgroup for Q6_K mat-vec.
                // NOTE: NR0 is compile-time in the Metal kernel. Only values that have a dedicated
                // kernel entry point are allowed here.
                // Usage: export GGML_METAL_Q6K_NR0=128
                const char * env_q6k_nr0 =
                is_decode_like ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q6K_NR0", "GGML_METAL_Q6K_NR0")
                                    : getenv("GGML_METAL_Q6K_NR0");
                 if (env_q6k_nr0) {
                   const int v = atoi(env_q6k_nr0);
                    if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128 || v == 256) {
                        nr0 = v;
                    }
                    static bool s_logged_q6k_nr0 = false;
                    if (!s_logged_q6k_nr0) {
                        s_logged_q6k_nr0 = true;
                        GGML_LOG_INFO("%s: Q6_K NR0 override = %d", __func__, (int) nr0);
                    }
                }
                
                // Safety clamp: avoid invalid (ne00 < nsg*nr0) combos.
                if (nsg < 1) {
                    nsg = 1;
                }
                const int max_nr0 = (int) (ne00 / nsg);
                if (max_nr0 >= 2) {
                    const int nr0_clamped = ggml_metal_nr0_allowed_leq_(max_nr0);
                    if (nr0 > nr0_clamped) {
                        nr0 = nr0_clamped;
                    }
                } else {
                    nsg = 1;
                    nr0 = 2;
                }
                // Optional visibility: log once when decode-like heuristics triggers.
                // This helps validate why GGML_METAL_DECODE_Q6K_* applies.
                static bool s_logged_decode_like = false;
                if (!s_logged_decode_like && is_decode_like && !is_decode_strict) {
                    s_logged_decode_like = true;
                    GGML_LOG_INFO("%s: decode-like mul_mv detected (ne11=%d ne12=%d ne13=%d); DECODE_* envs may apply\n",
                                  __func__, (int) ne11, (int) ne12, (int) ne13);
                }
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    // Q4_K/Q6_K NR0 is compile-time in the Metal kernel entrypoint name. If nr0 differs from
    // the default, we MUST pick the specialized entrypoint "_nr0_%d" (no silent fallback).
    if (tsrc0 == GGML_TYPE_Q4_K && nr0 != N_R0_Q4_K) {
        snprintf(base, 256, "kernel_mul_mv_%s_%s_nr0_%d%s",
                 ggml_type_name(tsrc0), ggml_type_name(tsrc1), nr0, suffix);
    } else if (tsrc0 == GGML_TYPE_Q6_K && nr0 != N_R0_Q6_K) {
        snprintf(base, 256, "kernel_mul_mv_%s_%s_nr0_%d%s",
                 ggml_type_name(tsrc0), ggml_type_name(tsrc1), nr0, suffix);
    } else {
        snprintf(base, 256, "kernel_mul_mv_%s_%s%s",
                 ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    }

    snprintf(name, 256, "%s_nsg=%d", base, nsg);
    
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;
    
    // Instrumentation (enable with -lv 4): show the effective base/name + params, and which envs
    // were eligible for *this* call (decode-like => DECODE_* takes precedence, else only ANY_*).
    if ((tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q6_K) && tsrc1 == GGML_TYPE_F32) {
        const char * dec_nsg = (tsrc0 == GGML_TYPE_Q4_K) ? getenv("GGML_METAL_DECODE_Q4K_NSG") : getenv("GGML_METAL_DECODE_Q6K_NSG");
        const char * dec_nr0 = (tsrc0 == GGML_TYPE_Q4_K) ? getenv("GGML_METAL_DECODE_Q4K_NR0") : getenv("GGML_METAL_DECODE_Q6K_NR0");
        const char * any_nsg = (tsrc0 == GGML_TYPE_Q4_K) ? getenv("GGML_METAL_Q4K_NSG")        : getenv("GGML_METAL_Q6K_NSG");
        const char * any_nr0 = (tsrc0 == GGML_TYPE_Q4_K) ? getenv("GGML_METAL_Q4K_NR0")        : getenv("GGML_METAL_Q6K_NR0");
        // Which envs were eligible for *this* call:
        // decode-like => DECODE_* takes precedence, else only ANY_*.
        const char * used_nsg = is_decode_like ? ((dec_nsg && dec_nsg[0]) ? dec_nsg : any_nsg) : any_nsg;
        const char * used_nr0 = is_decode_like ? ((dec_nr0 && dec_nr0[0]) ? dec_nr0 : any_nr0) : any_nr0;
        
        GGML_LOG_DEBUG("%s: %s mul_mv select: is_decode=%d base='%s' name='%s' | nr0=%d nr1=%d nsg=%d smem=%zu | used(nsg=%s nr0=%s) | env dec(nsg=%s nr0=%s) any(nsg=%s nr0=%s)\n",
                       __func__,
                       tsrc0 == GGML_TYPE_Q4_K ? "Q4_K" : "Q6_K",
                       (int) is_decode_like, base, name,
                       res.nr0, res.nr1, res.nsg, res.smem,
                       used_nsg ? used_nsg : "", used_nr0 ? used_nr0 : "",
                       dec_nsg ? dec_nsg : "", dec_nr0 ? dec_nr0 : "",
                       any_nsg ? any_nsg : "", any_nr0 ? any_nr0 : "");
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d", base, bc_inp);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    const bool is_decode_strict = (ne11 == 1 && ne12 == 1 && ne13 == 1);
    const int64_t aux_batch      = (int64_t) ne12 * (int64_t) ne13;
    const bool is_decode_like    = (ne11 == 1 && aux_batch <= 8);
    // NOTE:
    // MUL_MAT_ID is on the MoE hot path for Qwen3MoE. We must support the same NR0/NSG overrides
    // as MUL_MV (non-ID), but selecting kernel_mul_mv_id_* entry points.
 
    // The ID mat-vec kernel is specialized by NR0 at compile-time (Metal template parameter).
    // Only values with dedicated host_name entry points are allowed.
    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";
    // Decode-aware env overrides (prefer GGML_METAL_DECODE_* for decode-like mat-vec)
    const char * env_q4k_nsg = is_decode_like
        ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q4K_NSG", "GGML_METAL_Q4K_NSG")
        : getenv("GGML_METAL_Q4K_NSG");
    const char * env_q6k_nsg = is_decode_like
        ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q6K_NSG", "GGML_METAL_Q6K_NSG")
        : getenv("GGML_METAL_Q6K_NSG");
    
    const char * env_q4k_nr0 = is_decode_like
        ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q4K_NR0", "GGML_METAL_Q4K_NR0")
        : getenv("GGML_METAL_Q4K_NR0");
    const char * env_q6k_nr0 = is_decode_like
        ? ggml_metal_getenv_pref_("GGML_METAL_DECODE_Q6K_NR0", "GGML_METAL_Q6K_NR0")
        : getenv("GGML_METAL_Q6K_NR0");
    
    const bool has_q4k_nr0_env = (env_q4k_nr0 != NULL);
    const bool has_q6k_nr0_env = (env_q6k_nr0 != NULL);
    static bool s_logged_decode_like_id = false;
    if (!s_logged_decode_like_id && is_decode_like && !is_decode_strict) {
         s_logged_decode_like_id = true;
         GGML_LOG_INFO("%s: decode-like mul_mv_id detected (ne11=%d ne12=%d ne13=%d); DECODE_* envs may apply\n", __func__, (int) ne11, (int) ne12, (int) ne13);
    }
    
        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
                
                // Optional override: rows per threadgroup for F32/F16 mat-vec (non-short path).
                // This maps to args.nr0 consumed by the Metal kernel dispatch switch.
                // Usage:
                //   export GGML_METAL_F32_NR0=8
                //   export GGML_METAL_F16_NR0=8
                if (strcmp(suffix, "_short") != 0) {
                    const bool is_f32 = (tsrc0 == GGML_TYPE_F32);
                    const bool is_f16 = (tsrc0 == GGML_TYPE_F16);
                    const char * env = is_f32 ? getenv("GGML_METAL_F32_NR0")
                                     : is_f16 ? getenv("GGML_METAL_F16_NR0")
                              : NULL;
                    if (env) {
                                        const int v = atoi(env);
                        // Supported NR0 values are those compiled into the Metal dispatch switch.
                        if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32) {
                            nr0  = v;
                            smem = 32*sizeof(float)*nr0;
                        }
                        static bool s_logged_f32 = false;
                        static bool s_logged_f16 = false;
                        bool & s_logged = is_f32 ? s_logged_f32 : s_logged_f16;
                        if (!s_logged) {
                            s_logged = true;
                            GGML_LOG_INFO("%s: %s NR0 override = %d\n",
                                          __func__, is_f32 ? "F32" : "F16", (int) nr0);
                        }
                    }
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K: {
            nsg = N_SG_Q4_K;
            nr0 = N_R0_Q4_K;
            
            if (env_q4k_nsg && env_q4k_nsg[0]) {
                const int v = atoi(env_q4k_nsg);
                if (v >= 1 && v <= 8) {
                    nsg = v;
                }
            }
            
            if (env_q4k_nr0 && env_q4k_nr0[0]) {
               const int v_req = atoi(env_q4k_nr0);
              // Only values with dedicated entry points should be selected.
            // If the user asks something else, clamp down to the nearest allowed <= request.
                const int v = ggml_metal_nr0_allowed_leq_(v_req);
                if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128 || v == 256) {
                    nr0 = v;
                }
            }
            // Safety clamp: avoid invalid (ne00 < nsg*nr0) combos.
            if (nsg < 1) nsg = 1;
            if (nr0 < 2) nr0 = 2;
            
            const int max_nr0 = (int) (ne00 / nsg);
            if (max_nr0 >= 2) {
                const int nr0_clamped = ggml_metal_nr0_allowed_leq_(max_nr0);
                if (nr0 > nr0_clamped) {
                    nr0 = nr0_clamped;
                }
            } else {
                nsg = 1;
                nr0 = 2;
            }
                    } break;
            
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
                if (env_q6k_nsg) {
                    const int v = atoi(env_q6k_nsg);
                    if (v >= 1 && v <= 8) {
                        nsg = v;
                    }
                }

                if (env_q6k_nr0) {
                    const int v = atoi(env_q6k_nr0);
                    if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128 || v == 256) {
                        nr0 = v;
                    }
                }
            } break;
        case GGML_TYPE_IQ2_XXS:
            
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    // Q4_K/Q6_K NR0 is compile-time in the Metal kernel. If the user overrides NR0, the host must select
    // a matching kernel entry point (specialized by NR0) to avoid correctness issues.
    if (tsrc0 == GGML_TYPE_Q4_K && has_q4k_nr0_env && nr0 != N_R0_Q4_K) {
        // NR0 is a compile-time specialization. If the user requests an unsupported value,
        // clamp it to the nearest supported value <= requested (see ggml_metal_nr0_allowed_leq_()).
        const int v_req = nr0;
        const int v     = ggml_metal_nr0_allowed_leq_(v_req);
    
        nr0 = v;
    
        if (nr0 != N_R0_Q4_K) {
            snprintf(base, 256, "kernel_mul_mv_id_%s_%s_nr0_%d%s",
                     ggml_type_name(tsrc0), ggml_type_name(tsrc1), nr0, suffix);
        } else {
            snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s",
                     ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
        }
    } else if (tsrc0 == GGML_TYPE_Q6_K && has_q6k_nr0_env && nr0 != N_R0_Q6_K) {
        const int v = nr0;
        if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128 || v == 256) {
            snprintf(base, 256, "kernel_mul_mv_id_%s_%s_nr0_%d%s",
                     ggml_type_name(tsrc0), ggml_type_name(tsrc1), v, suffix);
        } else {
            // Unsupported override -> fall back to the default kernel and reset nr0 to the compiled default.
            nr0 = N_R0_Q6_K;
            snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s",
                     ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
        }
    } else {
        snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s",
                 ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    }

    snprintf(name, 256, "%s_nsg=%d", base, nsg);
    // Instrumentation (decode focus): confirm which Q4_K specialization is actually selected.
    // Enable with -lv 4 (GGML_LOG_DEBUG).
    if (tsrc0 == GGML_TYPE_Q4_K) {
        GGML_LOG_DEBUG(
            "%s: Q4_K mul_mv_id select: base='%s' name='%s' -> nr0=%d nsg=%d (env_nsg='%s' env_nr0='%s' has_nr0_env=%d)\n",
            __func__, base, name, nr0, nsg,
            env_q4k_nsg ? env_q4k_nsg : "",
            env_q4k_nr0 ? env_q4k_nr0 : "",
            (int) has_q4k_nr0_env);
    }
    
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;
    // Instrumentation: confirm Q4_K mul_mv_id uses the expected entry points and show which envs
    // were eligible for this call (decode-like => DECODE_* preferred, else only ANY_*).
    if (tsrc0 == GGML_TYPE_Q4_K && tsrc1 == GGML_TYPE_F32) {
        const char * env_dec_nsg = getenv("GGML_METAL_DECODE_Q4K_NSG");
        const char * env_dec_nr0 = getenv("GGML_METAL_DECODE_Q4K_NR0");
        const char * env_any_nsg = getenv("GGML_METAL_Q4K_NSG");
        const char * env_any_nr0 = getenv("GGML_METAL_Q4K_NR0");
        
        const char * used_nsg = is_decode_like ? ((env_dec_nsg && env_dec_nsg[0]) ? env_dec_nsg : env_any_nsg) : env_any_nsg;
        const char * used_nr0 = is_decode_like ? ((env_dec_nr0 && env_dec_nr0[0]) ? env_dec_nr0 : env_any_nr0) : env_any_nr0;
        GGML_LOG_DEBUG(
            "%s: Q4_K mul_mv_id select: is_decode=%d name='%s' | nr0=%d nr1=%d nsg=%d smem=%zu | used(nsg=%s nr0=%s) | env dec(nsg=%s nr0=%s) any(nsg=%s nr0=%s)\n",
            __func__, (int) is_decode_like, name, res.nr0, res.nr1, res.nsg, res.smem,
            used_nsg ? used_nsg : "", used_nr0 ? used_nr0 : "",
            env_dec_nsg ? env_dec_nsg : "", env_dec_nr0 ? env_dec_nr0 : "",
            env_any_nsg ? env_any_nsg : "", env_any_nr0 ? env_any_nr0 : "");
    }
     
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*(sizeof(float) + sizeof(int32_t));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// note: reuse the argsort kernel for top_k
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    // do bounds checks for the mask?
    const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_mask,
            ns10,
            ns20,
            nsg);
    if (const char * v = getenv("GGML_METAL_FA_LOG")) {
         if (v[0] && strcmp(v, "0") != 0) {
             GGML_LOG_INFO("ggml-metal: FA: pipeline %s\n", name);
         }
      }
    
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            ns10,
            ns20,
            nsg, nwg);
       if (const char * v = getenv("GGML_METAL_FA_LOG")) {
            if (v[0] && strcmp(v, "0") != 0) {
                GGML_LOG_INFO("ggml-metal: FA: pipeline %s\n", name);
            }
        }
    
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", base, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }
    GGML_UNUSED(op);
    return res;


}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(
        ggml_metal_library_t lib,
        ggml_op op,
        int32_t n_fuse,
        bool row) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD:   op_str = "add";   break;
        case GGML_OP_SUB:   op_str = "sub";   break;
        case GGML_OP_MUL:   op_str = "mul";   break;
        case GGML_OP_DIV:   op_str = "div";   break;
        default: GGML_ABORT("fatal error");
    };

    if (row) {
        snprintf(base, 256, "kernel_%s_row_c4_fuse_%d", op_str, n_fuse);
    } else {
        snprintf(base, 256, "kernel_%s_fuse_%d", op_str, n_fuse);
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    GGML_ASSERT(op->src[0]->ne[0] % 4 == 0);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_l2_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d", base, is_imrope ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_upscale_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }

    res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) {
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    int nsg = 1;
    while (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}
