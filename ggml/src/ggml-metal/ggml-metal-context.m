

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>
#include <pthread.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <inttypes.h>
#include <dlfcn.h>
#import "ggml-metal-context.h"

// Obj-C (.m) friendly helper: no C++ 'auto', no blocks.
static inline const char * ggml_metal_cmd_status_name(MTLCommandBufferStatus s) {
    switch (s) {
        case MTLCommandBufferStatusNotEnqueued: return "NotEnqueued";
        case MTLCommandBufferStatusEnqueued:    return "Enqueued";
        case MTLCommandBufferStatusCommitted:   return "Committed";
        case MTLCommandBufferStatusScheduled:   return "Scheduled";
        case MTLCommandBufferStatusCompleted:   return "Completed";
        case MTLCommandBufferStatusError:       return "Error";
        default:                                return "Unknown";
    }
}

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8


static inline bool ggml_metal_trace_copy_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = (getenv("GGML_METAL_TRACE_COPY") != NULL || getenv("GGML_METAL_MGPU_TRACE_COPY") != NULL) ? 1 : 0;
    }
    return cached == 1;
}


static inline bool ggml_metal_hot_tensor_name(const char * name) {
    if (!name || !name[0]) {
        return false;
    }
    return strcmp(name, "result_output") == 0 || strcmp(name, "embd") == 0 || strncmp(name, "l_out-", 6) == 0 || strcmp(name, " (copy)") == 0 || strncmp(name, "Metal#l_out-", 12) == 0 || strcmp(name, "Metal#embd#0") == 0 || strcmp(name, "Metal# (copy)#0") == 0;
}

enum { GGML_METAL_TRACE_CALLSITE_MAX = 128 };
static char     g_trace_callsite_tensor[GGML_METAL_TRACE_CALLSITE_MAX][64];
static char     g_trace_callsite_kind[GGML_METAL_TRACE_CALLSITE_MAX][24];
static char     g_trace_callsite_sym[GGML_METAL_TRACE_CALLSITE_MAX][128];
static uint64_t g_trace_callsite_calls[GGML_METAL_TRACE_CALLSITE_MAX];
static uint64_t g_trace_callsite_bytes[GGML_METAL_TRACE_CALLSITE_MAX];
static pthread_mutex_t g_trace_callsite_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline const char * ggml_metal_trace_callsite_symbol(void * ra, char * buf, size_t buf_size) {
    if (!buf || buf_size == 0) {
        return "";
    }
    buf[0] = '\0';
    if (!ra) {
        snprintf(buf, buf_size, "<null>");
        return buf;
    }
    Dl_info info;
    if (dladdr(ra, &info) && info.dli_sname && info.dli_sname[0]) {
        snprintf(buf, buf_size, "%s", info.dli_sname);
        return buf;
    }
    snprintf(buf, buf_size, "%p", ra);
    return buf;
}

static void ggml_metal_trace_callsite_record(const char * kind, const char * tensor_name, size_t size, void * ra) {
    if (!ggml_metal_trace_copy_enabled() || !ggml_metal_hot_tensor_name(tensor_name)) {
        return;
    }

    char sym[128];
    ggml_metal_trace_callsite_symbol(ra, sym, sizeof(sym));

    GGML_LOG_INFO("ggml-metal: trace-hot-callsite: kind=%s tensor='%s' bytes=%zu caller=%s\n",
        kind ? kind : "",
        tensor_name ? tensor_name : "",
        size,
        sym);

    pthread_mutex_lock(&g_trace_callsite_mutex);
    int free_slot = -1;
    for (int i = 0; i < GGML_METAL_TRACE_CALLSITE_MAX; ++i) {
        if (g_trace_callsite_tensor[i][0] == '\0') {
            if (free_slot < 0) {
                free_slot = i;
            }
            continue;
        }
        if (strncmp(g_trace_callsite_tensor[i], tensor_name ? tensor_name : "", sizeof(g_trace_callsite_tensor[i])) == 0 &&
            strncmp(g_trace_callsite_kind[i], kind ? kind : "", sizeof(g_trace_callsite_kind[i])) == 0 &&
            strncmp(g_trace_callsite_sym[i], sym, sizeof(g_trace_callsite_sym[i])) == 0) {
            g_trace_callsite_calls[i] += 1;
            g_trace_callsite_bytes[i] += size;
            pthread_mutex_unlock(&g_trace_callsite_mutex);
            return;
        }
    }
    if (free_slot >= 0) {
        snprintf(g_trace_callsite_tensor[free_slot], sizeof(g_trace_callsite_tensor[free_slot]), "%s", tensor_name ? tensor_name : "");
        snprintf(g_trace_callsite_kind[free_slot], sizeof(g_trace_callsite_kind[free_slot]), "%s", kind ? kind : "");
        snprintf(g_trace_callsite_sym[free_slot], sizeof(g_trace_callsite_sym[free_slot]), "%s", sym);
        g_trace_callsite_calls[free_slot] = 1;
        g_trace_callsite_bytes[free_slot] = size;
    }
    pthread_mutex_unlock(&g_trace_callsite_mutex);
}

static void ggml_metal_trace_callsite_summary(void) {
    if (!ggml_metal_trace_copy_enabled()) {
        return;
    }
    for (int rank = 0; rank < 10; ++rank) {
        int best = -1;
        for (int i = 0; i < GGML_METAL_TRACE_CALLSITE_MAX; ++i) {
            if (g_trace_callsite_tensor[i][0] == '\0') {
                continue;
            }
            if (best < 0 || g_trace_callsite_bytes[i] > g_trace_callsite_bytes[best]) {
                best = i;
            }
        }
        if (best < 0 || g_trace_callsite_bytes[best] == 0) {
            break;
        }
        GGML_LOG_INFO("ggml-metal: trace-hot-callsite-summary: rank=%d kind=%s tensor='%s' caller=%s calls=%llu bytes=%llu\n",
            rank + 1,
            g_trace_callsite_kind[best],
            g_trace_callsite_tensor[best],
            g_trace_callsite_sym[best],
            (unsigned long long) g_trace_callsite_calls[best],
            (unsigned long long) g_trace_callsite_bytes[best]);
        g_trace_callsite_tensor[best][0] = '\0';
    }
}



enum { GGML_METAL_TRACE_HOT_SEQ_MAX = 64 };
static char     g_trace_hot_seq_tensor[GGML_METAL_TRACE_HOT_SEQ_MAX][64];
static char     g_trace_hot_seq_kind[GGML_METAL_TRACE_HOT_SEQ_MAX][24];
static uint64_t g_trace_hot_seq_calls[GGML_METAL_TRACE_HOT_SEQ_MAX];
static uint64_t g_trace_hot_seq_bytes[GGML_METAL_TRACE_HOT_SEQ_MAX];
static pthread_mutex_t g_trace_hot_seq_mutex = PTHREAD_MUTEX_INITIALIZER;

static void ggml_metal_trace_hot_seq_record(const char * kind, const char * tensor_name, size_t size) {
    if (!ggml_metal_trace_copy_enabled() || !ggml_metal_hot_tensor_name(tensor_name)) {
        return;
    }

    uint64_t calls = 0;
    uint64_t total_bytes = 0;

    pthread_mutex_lock(&g_trace_hot_seq_mutex);
    int free_slot = -1;
    for (int i = 0; i < GGML_METAL_TRACE_HOT_SEQ_MAX; ++i) {
        if (g_trace_hot_seq_tensor[i][0] == '\0') {
            if (free_slot < 0) {
                free_slot = i;
            }
            continue;
        }
        if (strncmp(g_trace_hot_seq_tensor[i], tensor_name ? tensor_name : "", sizeof(g_trace_hot_seq_tensor[i])) == 0 &&
            strncmp(g_trace_hot_seq_kind[i], kind ? kind : "", sizeof(g_trace_hot_seq_kind[i])) == 0) {
            g_trace_hot_seq_calls[i] += 1;
            g_trace_hot_seq_bytes[i] += size;
            calls = g_trace_hot_seq_calls[i];
            total_bytes = g_trace_hot_seq_bytes[i];
            pthread_mutex_unlock(&g_trace_hot_seq_mutex);
            GGML_LOG_INFO("ggml-metal: trace-hot-seq: kind=%s tensor='%s' call=%llu bytes=%zu total_bytes=%llu\n",
                kind ? kind : "",
                tensor_name ? tensor_name : "",
                (unsigned long long) calls,
                size,
                (unsigned long long) total_bytes);
            return;
        }
    }
    if (free_slot >= 0) {
        snprintf(g_trace_hot_seq_tensor[free_slot], sizeof(g_trace_hot_seq_tensor[free_slot]), "%s", tensor_name ? tensor_name : "");
        snprintf(g_trace_hot_seq_kind[free_slot], sizeof(g_trace_hot_seq_kind[free_slot]), "%s", kind ? kind : "");
        g_trace_hot_seq_calls[free_slot] = 1;
        g_trace_hot_seq_bytes[free_slot] = size;
        calls = 1;
        total_bytes = size;
    }
    pthread_mutex_unlock(&g_trace_hot_seq_mutex);
    GGML_LOG_INFO("ggml-metal: trace-hot-seq: kind=%s tensor='%s' call=%llu bytes=%zu total_bytes=%llu\n",
        kind ? kind : "",
        tensor_name ? tensor_name : "",
        (unsigned long long) calls,
        size,
        (unsigned long long) total_bytes);
}

static void ggml_metal_trace_hot_seq_summary(void) {
    if (!ggml_metal_trace_copy_enabled()) {
        return;
    }
    for (int rank = 0; rank < 10; ++rank) {
        int best = -1;
        for (int i = 0; i < GGML_METAL_TRACE_HOT_SEQ_MAX; ++i) {
            if (g_trace_hot_seq_tensor[i][0] == '\0') {
                continue;
            }
            if (best < 0 || g_trace_hot_seq_bytes[i] > g_trace_hot_seq_bytes[best]) {
                best = i;
            }
        }
        if (best < 0 || g_trace_hot_seq_bytes[best] == 0) {
            break;
        }
        GGML_LOG_INFO("ggml-metal: trace-hot-seq-summary: rank=%d kind=%s tensor='%s' calls=%llu bytes=%llu\n",
            rank + 1,
            g_trace_hot_seq_kind[best],
            g_trace_hot_seq_tensor[best],
            (unsigned long long) g_trace_hot_seq_calls[best],
            (unsigned long long) g_trace_hot_seq_bytes[best]);
        g_trace_hot_seq_tensor[best][0] = '\0';
    }
}

static inline const char * ggml_metal_buf_label(id<MTLBuffer> buf) {
    if (buf == nil) {
        return "";
    }
    NSString * label = [buf label];
    return label ? [label UTF8String] : "";
}

#define GGML_METAL_TRACE_BUCKETS 7
#define GGML_METAL_TRACE_TOP_TENSORS 32
#define GGML_METAL_TRACE_NAME_MAX 64

static inline int ggml_metal_trace_size_bucket(size_t size) {
    if (size <= 4*1024)      return 0;
    if (size <= 16*1024)     return 1;
    if (size <= 64*1024)     return 2;
    if (size <= 256*1024)    return 3;
    if (size <= 1024*1024)   return 4;
    if (size <= 4*1024*1024) return 5;
    return 6;
}

static inline const char * ggml_metal_trace_bucket_name(int b) {
    switch (b) {
        case 0: return "le4k";
        case 1: return "4k_16k";
        case 2: return "16k_64k";
        case 3: return "64k_256k";
        case 4: return "256k_1m";
        case 5: return "1m_4m";
        default: return "gt4m";
    }
}

static inline int ggml_metal_trace_kind_index(const char * kind) {
    if (kind && strcmp(kind, "set_tensor_async") == 0) {
        return 0;
    }
    if (kind && strcmp(kind, "get_tensor_async") == 0) {
        return 1;
    }
    return 2;
}


struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;
    // decode scheduling instrumentation (opt-in via GGML_METAL_DECODE_STATS=1)
    bool     decode_stats_enable;
    bool     decode_stats_this_graph;
    uint64_t decode_stats_calls;
    uint64_t decode_stats_commits;
    uint64_t decode_stats_nodes_total;
    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // V5.1 Metal copy/sync tracing
    bool     trace_copy_enable;
    uint64_t trace_copy_blit_calls;
    uint64_t trace_copy_blit_bytes;
    uint64_t trace_copy_cross_dev_calls;
    uint64_t trace_copy_cross_dev_bytes;
    uint64_t trace_copy_sync_waits;
    uint64_t trace_copy_cmd_commits;
    uint64_t trace_copy_buffer_allocs;
    uint64_t trace_copy_buffer_alloc_bytes;
    uint64_t trace_copy_blit_calls_by_kind[3];
    uint64_t trace_copy_blit_bytes_by_kind[3];
    uint64_t trace_copy_blit_bucket_calls[GGML_METAL_TRACE_BUCKETS];
    uint64_t trace_copy_blit_bucket_bytes[GGML_METAL_TRACE_BUCKETS];
    uint64_t trace_copy_tensor_calls[GGML_METAL_TRACE_TOP_TENSORS];
    uint64_t trace_copy_tensor_bytes[GGML_METAL_TRACE_TOP_TENSORS];
    uint8_t  trace_copy_tensor_kind[GGML_METAL_TRACE_TOP_TENSORS];
    char     trace_copy_tensor_name[GGML_METAL_TRACE_TOP_TENSORS][GGML_METAL_TRACE_NAME_MAX];

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static inline const char * ggml_metal_trace_tensor_name(const char * name) {
    if (name == NULL || name[0] == '\0') {
        return "(unnamed)";
    }
    return name;
}

static inline void ggml_metal_trace_tensor_accum(ggml_metal_t ctx, const char * kind, const char * tensor_name, size_t size) {
    if (ctx == NULL) {
        return;
    }
    const int k = ggml_metal_trace_kind_index(kind);
    const char * safe = ggml_metal_trace_tensor_name(tensor_name);

    for (int i = 0; i < GGML_METAL_TRACE_TOP_TENSORS; ++i) {
        if (ctx->trace_copy_tensor_name[i][0] != '\0' &&
            ctx->trace_copy_tensor_kind[i] == (uint8_t) k &&
            strncmp(ctx->trace_copy_tensor_name[i], safe, GGML_METAL_TRACE_NAME_MAX) == 0) {
            ctx->trace_copy_tensor_calls[i] += 1;
            ctx->trace_copy_tensor_bytes[i] += size;
            return;
        }
    }

    for (int i = 0; i < GGML_METAL_TRACE_TOP_TENSORS; ++i) {
        if (ctx->trace_copy_tensor_name[i][0] == '\0') {
            strncpy(ctx->trace_copy_tensor_name[i], safe, GGML_METAL_TRACE_NAME_MAX - 1);
            ctx->trace_copy_tensor_name[i][GGML_METAL_TRACE_NAME_MAX - 1] = '\0';
            ctx->trace_copy_tensor_kind[i] = (uint8_t) k;
            ctx->trace_copy_tensor_calls[i] = 1;
            ctx->trace_copy_tensor_bytes[i] = size;
            return;
        }
    }

    int min_i = 0;
    for (int i = 1; i < GGML_METAL_TRACE_TOP_TENSORS; ++i) {
        if (ctx->trace_copy_tensor_bytes[i] < ctx->trace_copy_tensor_bytes[min_i]) {
            min_i = i;
        }
    }
    if (size > ctx->trace_copy_tensor_bytes[min_i]) {
        strncpy(ctx->trace_copy_tensor_name[min_i], safe, GGML_METAL_TRACE_NAME_MAX - 1);
        ctx->trace_copy_tensor_name[min_i][GGML_METAL_TRACE_NAME_MAX - 1] = '\0';
        ctx->trace_copy_tensor_kind[min_i] = (uint8_t) k;
        ctx->trace_copy_tensor_calls[min_i] = 1;
        ctx->trace_copy_tensor_bytes[min_i] = size;
    }
}

static inline const char * ggml_metal_trace_kind_name_from_index(int k) {
    switch (k) {
        case 0: return "set_tensor_async";
        case 1: return "get_tensor_async";
        default: return "other";
    }
}

static inline void ggml_metal_trace_emit_top_tensors(ggml_metal_t ctx) {
    uint64_t bytes[GGML_METAL_TRACE_TOP_TENSORS];
    uint64_t calls[GGML_METAL_TRACE_TOP_TENSORS];
    uint8_t  kind[GGML_METAL_TRACE_TOP_TENSORS];
    char     names[GGML_METAL_TRACE_TOP_TENSORS][GGML_METAL_TRACE_NAME_MAX];
    memcpy(bytes, ctx->trace_copy_tensor_bytes, sizeof(bytes));
    memcpy(calls, ctx->trace_copy_tensor_calls, sizeof(calls));
    memcpy(kind,  ctx->trace_copy_tensor_kind,  sizeof(kind));
    memcpy(names, ctx->trace_copy_tensor_name,  sizeof(names));

    for (int rank = 0; rank < 10; ++rank) {
        int best = -1;
        for (int i = 0; i < GGML_METAL_TRACE_TOP_TENSORS; ++i) {
            if (names[i][0] == '\0') {
                continue;
            }
            if (best < 0 || bytes[i] > bytes[best]) {
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        GGML_LOG_INFO("ggml-metal: trace-top-tensor: rank=%d kind=%s name='%s' calls=%llu bytes=%llu\n",
            rank + 1,
            ggml_metal_trace_kind_name_from_index(kind[best]),
            names[best],
            (unsigned long long) calls[best],
            (unsigned long long) bytes[best]);
        names[best][0] = '\0';
    }

    ggml_metal_trace_callsite_summary();
    ggml_metal_trace_hot_seq_summary();
}

static inline void ggml_metal_trace_blit(ggml_metal_t ctx, const char * tag, const char * kind, const char * tensor_name, id<MTLBuffer> src, id<MTLBuffer> dst, size_t src_off, size_t dst_off, size_t size) {
    if (!ggml_metal_trace_copy_enabled()) {
        return;
    }
    const char * src_dev = src ? [[[src device] name] UTF8String] : "";
    const char * dst_dev = dst ? [[[dst device] name] UTF8String] : "";
    GGML_LOG_INFO("ggml-metal: %s: kind=%s tensor='%s' src_buf=%p dst_buf=%p src_dev=%s dst_dev=%s src_label='%s' dst_label='%s' src_off=%zu dst_off=%zu size=%zu bucket=%s\n",
        tag, kind ? kind : "unknown", ggml_metal_trace_tensor_name(tensor_name), (void *) src, (void *) dst, src_dev, dst_dev, ggml_metal_buf_label(src), ggml_metal_buf_label(dst), src_off, dst_off, size, ggml_metal_trace_bucket_name(ggml_metal_trace_size_bucket(size)));
    if (ctx) {
        const int b = ggml_metal_trace_size_bucket(size);
        const int k = ggml_metal_trace_kind_index(kind);
        ctx->trace_copy_blit_calls += 1;
        ctx->trace_copy_blit_bytes += size;
        ctx->trace_copy_blit_bucket_calls[b] += 1;
        ctx->trace_copy_blit_bucket_bytes[b] += size;
        ctx->trace_copy_blit_calls_by_kind[k] += 1;
        ctx->trace_copy_blit_bytes_by_kind[k] += size;
        ggml_metal_trace_tensor_accum(ctx, kind, tensor_name, size);
        if (src && dst && [src device] != [dst device]) {
            ctx->trace_copy_cross_dev_calls += 1;
            ctx->trace_copy_cross_dev_bytes += size;
        }
    }
}


ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    if (dev == nil) {
          GGML_LOG_ERROR("ggml-metal: ggml_metal_init called with dev=NULL\n");
         return NULL;
    }
    
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> d in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[d name] UTF8String]);
    }
    [devices release]; // created by a *Copy* method
#endif

    ggml_metal_t res = (ggml_metal_t) calloc(1, sizeof(struct ggml_metal));
    if (!res) {
        GGML_LOG_ERROR("%s: error: failed to allocate ggml_metal\n", __func__);
        return NULL;
    }
    
    // Critical: persist the selected device in the context (used by graph_compute and others)
    res->dev = dev;
    
    GGML_LOG_INFO("%s: ctx=%p dev=%p\n", __func__, (void *) res, (void *) res->dev); //Debug Log TOTO
    
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    GGML_LOG_INFO("%s: using requested device: %s\n", __func__, [[device name] UTF8String]);

    // Use per-device queue (your design choice)
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        free(res);
        return NULL;
    }
    res->cmd_bufs_ext = [[NSMutableArray alloc] init];
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);
            free(res);
            return NULL;
        }
    }

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == NULL;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == NULL;

    // On discrete GPUs (non-unified), concurrency may cause non-deterministic results.
    // Default to disabling it unless explicitly forced.
    {
        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);
        const bool force_concurrency = getenv("GGML_METAL_CONCURRENCY_FORCE") != NULL;

        if (!force_concurrency && props_dev && !props_dev->has_unified_memory) {
            if (res->use_concurrency) {
                GGML_LOG_WARN("%s: disabling concurrency on discrete (non-unified) GPU; set GGML_METAL_CONCURRENCY_FORCE=1 to override\n", __func__);
            }
            res->use_concurrency = false;
        }
    }

    // Debug flags:
    // - GGML_METAL_GRAPH_DEBUG / GGML_METAL_FUSION_DEBUG : fine-grained toggles
    // - GGML_METAL_DEBUG : umbrella switch enabling both
    const bool debug_all = getenv("GGML_METAL_DEBUG") != NULL;

    const char * val_graph = getenv("GGML_METAL_GRAPH_DEBUG");
    res->debug_graph  = debug_all ? 1 : (val_graph  ? atoi(val_graph)  : 0);

    const char * val_fusion = getenv("GGML_METAL_FUSION_DEBUG");
    res->debug_fusion = debug_all ? 1 : (val_fusion ? atoi(val_fusion) : 0);

    res->decode_stats_enable     = getenv("GGML_METAL_DECODE_STATS") != NULL;
    res->decode_stats_this_graph = false;
    res->decode_stats_calls      = 0;
    res->decode_stats_commits    = 0;
    res->decode_stats_nodes_total = 0;

    res->trace_copy_enable = ggml_metal_trace_copy_enabled();
    res->trace_copy_blit_calls = 0;
    res->trace_copy_blit_bytes = 0;
    res->trace_copy_cross_dev_calls = 0;
    res->trace_copy_cross_dev_bytes = 0;
    res->trace_copy_sync_waits = 0;
    res->trace_copy_cmd_commits = 0;
    res->trace_copy_buffer_allocs = 0;
    res->trace_copy_buffer_alloc_bytes = 0;
    memset(res->trace_copy_blit_calls_by_kind, 0, sizeof(res->trace_copy_blit_calls_by_kind));
    memset(res->trace_copy_blit_bytes_by_kind, 0, sizeof(res->trace_copy_blit_bytes_by_kind));
    memset(res->trace_copy_blit_bucket_calls, 0, sizeof(res->trace_copy_blit_bucket_calls));
    memset(res->trace_copy_blit_bucket_bytes, 0, sizeof(res->trace_copy_blit_bucket_bytes));
    memset(res->trace_copy_tensor_calls, 0, sizeof(res->trace_copy_tensor_calls));
    memset(res->trace_copy_tensor_bytes, 0, sizeof(res->trace_copy_tensor_bytes));
    memset(res->trace_copy_tensor_kind, 0, sizeof(res->trace_copy_tensor_kind));
    memset(res->trace_copy_tensor_name, 0, sizeof(res->trace_copy_tensor_name));

    // Graph optimizer toggle (default ON)
    res->use_graph_optimize = true;
    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    // Init counters used by fusion stats, etc.
    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");
    GGML_LOG_INFO("%s: debug graph        = %d\n", __func__, res->debug_graph);
    GGML_LOG_INFO("%s: debug fusion       = %d\n", __func__, res->debug_fusion);
    GGML_LOG_INFO("%s: decode stats       = %s\n", __func__, res->decode_stats_enable ? "true" : "false");

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->trace_copy_enable) {
        GGML_LOG_INFO("ggml-metal: trace-summary: blit_calls=%llu blit_bytes=%llu cross_dev_calls=%llu cross_dev_bytes=%llu sync_waits=%llu cmd_commits=%llu buffer_allocs=%llu buffer_alloc_bytes=%llu\n",
            (unsigned long long) ctx->trace_copy_blit_calls,
            (unsigned long long) ctx->trace_copy_blit_bytes,
            (unsigned long long) ctx->trace_copy_cross_dev_calls,
            (unsigned long long) ctx->trace_copy_cross_dev_bytes,
            (unsigned long long) ctx->trace_copy_sync_waits,
            (unsigned long long) ctx->trace_copy_cmd_commits,
            (unsigned long long) ctx->trace_copy_buffer_allocs,
            (unsigned long long) ctx->trace_copy_buffer_alloc_bytes);
        GGML_LOG_INFO("ggml-metal: trace-kind-summary: set_calls=%llu set_bytes=%llu get_calls=%llu get_bytes=%llu other_calls=%llu other_bytes=%llu\n",
            (unsigned long long) ctx->trace_copy_blit_calls_by_kind[0],
            (unsigned long long) ctx->trace_copy_blit_bytes_by_kind[0],
            (unsigned long long) ctx->trace_copy_blit_calls_by_kind[1],
            (unsigned long long) ctx->trace_copy_blit_bytes_by_kind[1],
            (unsigned long long) ctx->trace_copy_blit_calls_by_kind[2],
            (unsigned long long) ctx->trace_copy_blit_bytes_by_kind[2]);
        for (int bi = 0; bi < GGML_METAL_TRACE_BUCKETS; ++bi) {
            GGML_LOG_INFO("ggml-metal: trace-bucket-summary: bucket=%s calls=%llu bytes=%llu\n",
                ggml_metal_trace_bucket_name(bi),
                (unsigned long long) ctx->trace_copy_blit_bucket_calls[bi],
                (unsigned long long) ctx->trace_copy_blit_bucket_bytes[bi]);
        }
        ggml_metal_trace_emit_top_tensors(ctx);
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    free(ctx);
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        const MTLCommandBufferStatus status = [ctx->cmd_buf_last status];
        if (status != MTLCommandBufferStatusNotEnqueued && status != MTLCommandBufferStatusCompleted) {
            if (ctx->trace_copy_enable) {
                ctx->trace_copy_sync_waits += 1;
                GGML_LOG_INFO("ggml-metal: cmd-buffer-wait: scope=cmd_buf_last cmd=%p status=%s\n", (void *) ctx->cmd_buf_last, ggml_metal_cmd_status_name(status));
            }
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        ctx->cmd_buf_last = nil;
    }

    const bool no_abort = getenv("GGML_METAL_SYNC_NO_ABORT") != NULL;

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) continue;

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status == MTLCommandBufferStatusNotEnqueued) continue;

            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: cb[%d]=%p status(before)=%d(%s)\n", __func__, cb_idx, (void *) cmd_buf, (int) status, ggml_metal_cmd_status_name(status));
                if (ctx->trace_copy_enable) {
                    ctx->trace_copy_sync_waits += 1;
                    GGML_LOG_INFO("ggml-metal: cmd-buffer-wait: scope=cb[%d] cmd=%p status=%s\n", cb_idx, (void *) cmd_buf, ggml_metal_cmd_status_name(status));
                }
                if (ctx->trace_copy_enable) {
                    ctx->trace_copy_sync_waits += 1;
                    GGML_LOG_INFO("ggml-metal: cmd-buffer-wait: scope=cb[%d]-ext cmd=%p status=%s\n", cb_idx, (void *) cmd_buf, ggml_metal_cmd_status_name(status));
                }
                [cmd_buf waitUntilCompleted];
                status = [cmd_buf status];
                GGML_LOG_ERROR("%s: cb[%d]=%p status(after) =%d(%s)\n", __func__, cb_idx, (void *) cmd_buf, (int) status, ggml_metal_cmd_status_name(status));
            }

            if (status != MTLCommandBufferStatusCompleted) {
                NSError * err = [cmd_buf error];
                GGML_LOG_ERROR("%s: error: command buffer %d did not complete (status=%d/%s)\n",
                                __func__, cb_idx, (int) status, ggml_metal_cmd_status_name(status));
                if (err) {
                    GGML_LOG_ERROR("%s: cb[%d] NSError domain=%s code=%ld desc=%s\n",
                                   __func__, cb_idx,
                                   [[err domain] UTF8String],
                                   (long) [err code],
                                   [[err localizedDescription] UTF8String]);
                }

                if (no_abort) {
                    GGML_LOG_ERROR("%s: GGML_METAL_SYNC_NO_ABORT=1 -> returning early (non-fatal)\n", __func__);
                    return;
                }
                GGML_ABORT("fatal error");
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status == MTLCommandBufferStatusNotEnqueued) {
                [cmd_buf release];
                continue;
            }

            if (status != MTLCommandBufferStatusCompleted) {
                [cmd_buf waitUntilCompleted];
                status = [cmd_buf status];
            }

            if (status != MTLCommandBufferStatusCompleted) {
                NSError * err = [cmd_buf error];
                GGML_LOG_ERROR("%s: error: ext command buffer %zu did not complete (status=%d/%s)\n",
                                __func__, i, (int) status, ggml_metal_cmd_status_name(status));
                if (err) {
                    GGML_LOG_ERROR("%s: ext[%zu] NSError domain=%s code=%ld desc=%s\n",
                                   __func__, i,
                                   [[err domain] UTF8String],
                                   (long) [err code],
                                   [[err localizedDescription] UTF8String]);
                }

                if (no_abort) {
                    GGML_LOG_ERROR("%s: GGML_METAL_SYNC_NO_ABORT=1 -> continuing cleanup (non-fatal)\n", __func__);
                    [cmd_buf release];
                    continue;
                }
                GGML_ABORT("fatal error");
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                         length:size
                                                        options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);
        if (ctx->trace_copy_enable) {
            ctx->trace_copy_buffer_allocs += 1;
            ctx->trace_copy_buffer_alloc_bytes += size;
            GGML_LOG_INFO("ggml-metal: buffer-alloc: kind=set_tensor_async bytes=%zu storage=%u buf=%p dev=%s\n",
                size, (unsigned) MTLResourceStorageModeShared, (void *) buf_src, [[device name] UTF8String]);
        }

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }
        id<MTLDevice> dst_dev = [(id<MTLBuffer>) bid_dst.metal device];
        if (dst_dev != device) {
            GGML_LOG_ERROR("%s: device mismatch for tensor '%s': ctx_dev=%p(%s) dst_buf_dev=%p(%s)\n",
                           __func__, tensor->name,
                           (void *) device, [[device name] UTF8String],
                           (void *) dst_dev, [[dst_dev name] UTF8String]);
            GGML_ABORT("fatal error");        bid_dst.offs += offset;
        }
        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        if (ctx->trace_copy_enable) {
            GGML_LOG_INFO("ggml-metal: cmd-buffer-create: scope=set_tensor_async cmd=%p\n", (void *) cmd_buf);
        }

        ggml_metal_trace_callsite_record("set_tensor_async", tensor ? tensor->name : "", size, __builtin_return_address(0));
        ggml_metal_trace_hot_seq_record("set_tensor_async", tensor ? tensor->name : "", size);
        ggml_metal_trace_blit(ctx, "blit-copy", "set_tensor_async", tensor ? tensor->name : "", buf_src, bid_dst.metal, 0, bid_dst.offs, size);
        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        if (ctx->trace_copy_enable) {
            ctx->trace_copy_cmd_commits += 1;
            GGML_LOG_INFO("ggml-metal: cmd-buffer-commit: scope=set_tensor_async cmd=%p\n", (void *) cmd_buf);
        }
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);
        if (ctx->trace_copy_enable) {
            ctx->trace_copy_buffer_allocs += 1;
            ctx->trace_copy_buffer_alloc_bytes += size;
            GGML_LOG_INFO("ggml-metal: buffer-alloc: kind=get_tensor_async bytes=%zu storage=%u buf=%p dev=%s\n",
                size, (unsigned) MTLResourceStorageModeShared, (void *) buf_dst, [[device name] UTF8String]);
        }

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        if (ctx->trace_copy_enable) {
            GGML_LOG_INFO("ggml-metal: cmd-buffer-create: scope=get_tensor_async cmd=%p\n", (void *) cmd_buf);
        }

        ggml_metal_trace_callsite_record("get_tensor_async", tensor ? tensor->name : "", size, __builtin_return_address(0));
        ggml_metal_trace_hot_seq_record("get_tensor_async", tensor ? tensor->name : "", size);
        ggml_metal_trace_blit(ctx, "blit-copy", "get_tensor_async", tensor ? tensor->name : "", bid_src.metal, buf_dst, bid_src.offs, 0, size);
        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        if (ctx->trace_copy_enable) {
            ctx->trace_copy_cmd_commits += 1;
            GGML_LOG_INFO("ggml-metal: cmd-buffer-commit: scope=get_tensor_async cmd=%p\n", (void *) cmd_buf);
        }
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}
// decode: force single CB/CE option via env GGML_METAL_DECODE_1CB=1
enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 64;

    // number of worker command buffers (in addition to the main-thread buffer)
    // note: the main-thread buffer is stored at index ctx->n_cb (see original scheme below)
    const int cb_main = ctx->n_cb;
    int n_cb_workers  = ctx->n_cb;
    
    // Decode scheduling knob (opt-in):
    //
    // Usage:
    //  export GGML_METAL_DECODE_SCHED=1
    //  export GGML_METAL_DECODE_NODES_MAX=256   (optional, default 256)
    //
    // If enabled, we collapse small/token-like graphs to a single CB/CE by setting
    // n_cb_workers=0 while keeping cb_main index unchanged.
    ctx->decode_stats_this_graph = false;
    
    if (getenv("GGML_METAL_DECODE_SCHED") != NULL) {
        int nodes_max = 256;
        const char * env = getenv("GGML_METAL_DECODE_NODES_MAX");
        if (env && env[0]) {
            const int v = atoi(env);
            if (v > 0) {
                nodes_max = v;
            }
        }
        // decode workers limit: acts as a cap when combined with MIN logic
        int decode_workers = 0;
        env = getenv("GGML_METAL_DECODE_CB_WORKERS");
        if (env && env[0]) {
            const int v = atoi(env);
            if (v >= 0) {
                decode_workers = v;
            }
            
            if (gf->n_nodes <= nodes_max) {
                // Keep cb_main index unchanged; only adjust worker count.
                // If GGML_METAL_DECODE_CB_WORKERS is unset, decode_workers defaults to 0
                // which collapses decode-like graphs to a single CB/CE.
                n_cb_workers = MIN(n_cb_workers, decode_workers);
                ctx->decode_stats_this_graph = true;
            }
        }
        // decode stats: accumulate per-graph counters only when the decode heuristic triggers
        if (ctx->decode_stats_enable && ctx->decode_stats_this_graph) {
             ctx->decode_stats_calls++;
             ctx->decode_stats_nodes_total += (uint64_t) gf->n_nodes;
        }
    }
    if (ctx->dev == nil) {
         GGML_LOG_ERROR("ggml-metal: ctx->dev is NULL before rsets_keep_alive\n");
          // ggml_metal_graph_compute is non-void: fail fast instead of segfaulting
         return GGML_STATUS_FAILED;
      }
    
    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        // number of nodes per worker CB (avoid division by zero when workers are disabled)
        if (n_cb_workers > 0) {
            const int n_cb_denom = n_cb_workers;
            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + n_cb_denom - 1) / n_cb_denom;
        } else {
            // all remaining nodes will be encoded by cb_main
            ctx->n_nodes_per_cb = ctx->n_nodes_1;
        }
        const bool use_capture = ctx->capture_next_compute;
        if (use_capture) {
            ctx->capture_next_compute = false;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithFormat:@"/tmp/perf-metal.gputrace"]];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[cb_main]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_main].obj) {
                [ctx->cmd_bufs[cb_main].obj release];
            }
            ctx->cmd_bufs[cb_main].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(cb_main);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[cb_main].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb_workers)
        for (int cb_idx = 0; cb_idx < n_cb_workers; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb_workers, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (!use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_main].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, cb_main, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb_workers; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb_workers ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }
            
            // decode stats: count commits originating from graph_compute path
            if (ctx->decode_stats_enable && ctx->decode_stats_this_graph) {
                ctx->decode_stats_commits++;
                if ((ctx->decode_stats_calls % 200) == 0) {
                    const double avg_nodes = (double) ctx->decode_stats_nodes_total / (double) (ctx->decode_stats_calls ? ctx->decode_stats_calls : 1);
                    const double commits_per_call = (double) ctx->decode_stats_commits / (double) (ctx->decode_stats_calls ? ctx->decode_stats_calls : 1);
                    GGML_LOG_INFO("metal decode stats: calls=%" PRIu64 " avg_nodes=%.1f commits/call=%.2f\n",
                                  ctx->decode_stats_calls, avg_nodes, commits_per_call);
                }
            }
            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_next_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
            
            // decode stats: count commits originating from encode_async workers
            if (ctx->decode_stats_enable && ctx->decode_stats_this_graph) {
                ctx->decode_stats_commits++;
                // log periodically from cb0 to avoid spamming
                if (cb_idx == 0 && (ctx->decode_stats_calls % 200) == 0) {
                    const double avg_nodes = (double) ctx->decode_stats_nodes_total / (double) (ctx->decode_stats_calls ? ctx->decode_stats_calls : 1);
                    const double commits_per_call = (double) ctx->decode_stats_commits / (double) (ctx->decode_stats_calls ? ctx->decode_stats_calls : 1);
                    GGML_LOG_INFO("metal decode stats: calls=%" PRIu64 " avg_nodes=%.1f commits/call=%.2f\n",
                                  ctx->decode_stats_calls, avg_nodes, commits_per_call);
                }
            }
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_next_compute = true;
}
