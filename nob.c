#define NOB_IMPLEMENTATION
#include "nob.h"
#include <stdlib.h>
#include <string.h>

const char* env_or_default(const char* key, const char* def) {
    const char* val = getenv(key);
    return (val && val[0] != '\0') ? val : def;
}

void load_dotenv(void) {
    if(!nob_file_exists(".env")) return;

    Nob_String_Builder sb = {0};
    if(!nob_read_entire_file(".env", &sb)) return;

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
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_SEQUENCES=%s", env_or_default("MAX_SEQUENCES", "4")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_PROMPT_LEN=%s", env_or_default("MAX_PROMPT_LEN", "512")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_SEQ_LEN=%s", env_or_default("MAX_SEQ_LEN", "2048")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_MAX_NUM_THREAD=%s", env_or_default("MAX_NUM_THREAD", "1024")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_TOP_K=%s", env_or_default("DEFAULT_TOP_K", "40")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_TEMPERATURE=%sf", env_or_default("DEFAULT_TEMPERATURE", "0.8")));
    nob_cmd_append(cmd, nob_temp_sprintf("-DENGINE_DEFAULT_MAX_NEW_TOKENS=%s", env_or_default("DEFAULT_MAX_NEW_TOKENS", "20")));
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

    load_dotenv();

    const char* cuda_path = resolve_cuda_path();
    nob_log(NOB_INFO, "Using cuda path: %s", cuda_path);

    if(!nob_mkdir_if_not_exists("build")) return 1;

    const char* arch = env_or_default("ARCH", "native");
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