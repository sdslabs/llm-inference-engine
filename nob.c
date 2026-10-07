#define NOB_IMPLEMENTATION
#include "nob.h"
#include <stdlib.h>
#include <string.h>

static const char* REQUIRED_KEYS[] = {
    "ARCH",
    "MAX_SEQUENCES",
    "MAX_PROMPT_LEN",
    "MAX_SEQ_LEN",
    "MAX_NUM_THREAD",
    "MAX_TOP_K",
    "DEFAULT_TOP_K",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_NEW_TOKENS",
};

int check_required_env(void) {
    int missing = 0;
    for(size_t i = 0; i < NOB_ARRAY_LEN(REQUIRED_KEYS); i++) {
        const char* val = getenv(REQUIRED_KEYS[i]);
        if(!val || val[0] == '\0') {
            nob_log(NOB_ERROR, "%s is not set", REQUIRED_KEYS[i]);
            missing++;
        }
    }
    return missing;
}

int load_dotenv(void) {
    if(!nob_file_exists(".env")) {
        nob_log(NOB_ERROR, ".env not found; it is the only source for the build limits");
        return 0;
    }

    Nob_String_Builder sb = {0};
    if(!nob_read_entire_file(".env", &sb)) return 0;

    Nob_String_View content = {.count=sb.count, .data = sb.items};

    while(content.count > 0) {
        // This grabs one line at a time
        Nob_String_View line = nob_sv_chop_by_delim(&content, '\n');

        // Strips .env comments
        line = nob_sv_trim(nob_sv_chop_by_delim(&line, '#'));
        if(line.count == 0) continue;

        Nob_String_View key = nob_sv_trim(nob_sv_chop_by_delim(&line, '='));
        Nob_String_View val = nob_sv_trim(line);
        
        if(key.count > 0) {
            // nob_temp_sprintf uses the arena allocator to yield null-terminated strings
            const char* key_cstr = nob_temp_sprintf(SV_Fmt, SV_Arg(key));
            const char* val_cstr = nob_temp_sprintf(SV_Fmt, SV_Arg(val));
            
            // setenv with overwrite=0 so existing env vars still take priority
            setenv(key_cstr, val_cstr, 0);
        }
    }

    nob_sb_free(sb);
    return 1;
}

const char* resolve_cuda_path(void) {
    const char* env_path = getenv("CUDA_PATH");
    if(env_path && strlen(env_path) > 0) return env_path;

    if(nob_file_exists("/usr/local/cuda")) return "/usr/local/cuda";
    if(nob_file_exists("/opt/cuda")) return "/opt/cuda";

    // Fallback
    return "/usr/local/cuda";
}

void append_limits(Nob_Cmd* cmd) {
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_SEQUENCES=%s", getenv("MAX_SEQUENCES")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_PROMPT_LEN=%s", getenv("MAX_PROMPT_LEN")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_SEQ_LEN=%s", getenv("MAX_SEQ_LEN")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_NUM_THREAD=%s", getenv("MAX_NUM_THREAD")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_TOP_K=%s", getenv("MAX_TOP_K")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_TOP_K=%s", getenv("DEFAULT_TOP_K")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_TEMPERATURE=%sf", getenv("DEFAULT_TEMPERATURE")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_MAX_NEW_TOKENS=%s", getenv("DEFAULT_MAX_NEW_TOKENS")));
}

int main(int argc, char** argv) {
    NOB_GO_REBUILD_URSELF(argc, argv);

    if(argc > 1 && strcmp(argv[1], "clean") == 0) {
        Nob_Cmd cmd = {0};
        nob_cmd_append(&cmd, "rm", "-rf", "build");
        nob_cmd_run_sync(cmd);
        nob_log(NOB_INFO, "cleaned");
        return 0;
    }

    if(!load_dotenv()) return 1;
    if(check_required_env() > 0) {
        nob_log(NOB_ERROR, "define the missing keys in .env, or export them to override");
        return 1;
    }

    const char* cuda_path = resolve_cuda_path();
    nob_log(NOB_INFO, "Using cuda path: %s", cuda_path);

    if(!nob_mkdir_if_not_exists("build")) return 1;

    const char* arch = getenv("ARCH");
    const char* tokenizer_dir = "external/tokenizers-cpp";

    // Compile CUDA kernals
    Nob_Cmd nvcc = {0};
    nob_cmd_append(&nvcc, "nvcc", "-O3", nob_temp_sprintf("-arch=%s", arch));
    append_limits(&nvcc);
    nob_cmd_append(&nvcc, "-c", "src/kernels.cu", "-o", "build/kernels.o");

    nob_log(NOB_INFO, "Compiling CUDA Kernels for %s ", arch);
    if(!nob_cmd_run_sync(nvcc)) return 1;

    // Compile Main Binary
    Nob_Cmd gpp = {0};
    nob_cmd_append(&gpp, "g++", "-O3", "-std=c++17");
    append_limits(&gpp);

    // Inputs
    nob_cmd_append(&gpp,
        "src/main.cpp", "src/config.cpp", "src/runtime.cpp",
        "src/model.cpp", "src/score.cpp", "src/telemetry.cpp",
        "build/kernels.o"
    );

    // Output
    nob_cmd_append(&gpp, "-o", "build/engine");

    // Includes
    nob_cmd_append(&gpp, "-Isrc");
    nob_cmd_append(&gpp, nob_temp_sprintf("-I%s/include", tokenizer_dir));
    nob_cmd_append(&gpp, nob_temp_sprintf("-I%s/include", cuda_path));

    // Libraries
    nob_cmd_append(&gpp, nob_temp_sprintf("-L%s/build", tokenizer_dir), "-ltokenizers_cpp", "-ltokenizers_c");
    nob_cmd_append(&gpp, nob_temp_sprintf("-L%s/lib64", cuda_path), "-lcudart", "-lcublas");

    nob_log(NOB_INFO, "Compiling Main Binary..");
    if(!nob_cmd_run_sync(gpp)) return 1;

    nob_log(NOB_INFO, "Build complete : ./build/engine");
    return 0;
}