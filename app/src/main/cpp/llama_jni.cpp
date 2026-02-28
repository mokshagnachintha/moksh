#include <jni.h>
#include <string>
#include <vector>
#include <ctime>
#include <android/log.h>
#include "llama.h"

#define TAG  "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ── Global state (single model + context) ─────────────────────────────────
static llama_model   * g_model = nullptr;
static llama_context * g_ctx   = nullptr;
// ──────────────────────────────────────────────────────────────────────────

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_orag_ai_LlamaBridge_loadModel(JNIEnv *env, jobject /*thiz*/, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model from: %s", path);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU-only on Android

    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    // Cap at 4 threads on mobile: avoids thermal throttling on big.LITTLE chips
    // and prevents the OS from killing the process under memory pressure.
    // 4 is the sweet-spot recommended for on-device llama.cpp inference.
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    LOGI("Model loaded successfully. Vocab: %d, Embed: %d",
         llama_vocab_n_tokens(vocab), llama_model_n_embd(g_model));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_orag_ai_LlamaBridge_unloadModel(JNIEnv */*env*/, jobject /*thiz*/) {
    if (g_ctx)   { llama_free(g_ctx);          g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model);  g_model = nullptr; }
    llama_backend_free();
    LOGI("Model unloaded");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_orag_ai_LlamaBridge_generateResponse(JNIEnv *env, jobject /*thiz*/, jstring jprompt) {
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("[Error: model not loaded]");
    }

    const char *raw = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt(raw);
    env->ReleaseStringUTFChars(jprompt, raw);

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    // ── Tokenise ────────────────────────────────────────────────────────────
    std::vector<llama_token> tokens(2048);
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(), (int32_t)prompt.size(),
        tokens.data(), (int32_t)tokens.size(),
        /*add_special=*/true,
        /*parse_special=*/true
    );
    if (n_tokens < 0) {
        return env->NewStringUTF("[Error: tokenisation failed]");
    }
    tokens.resize(n_tokens);

    // ── Decode prompt ───────────────────────────────────────────────────────
    llama_kv_self_clear(g_ctx);
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(g_ctx, batch) != 0) {
        return env->NewStringUTF("[Error: prompt decode failed]");
    }

    // ── Sampler chain ───────────────────────────────────────────────────────
    llama_sampler *sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    // Use current time as seed so each call produces different output
    llama_sampler_chain_add(sampler, llama_sampler_init_dist((uint32_t)time(nullptr)));

    // ── Auto-regressive generation ──────────────────────────────────────────
    const llama_token eos_token = llama_vocab_eos(vocab);
    std::string result;
    result.reserve(512);
    char piece_buf[256];

    for (int i = 0; i < 512; ++i) {
        llama_token new_tok = llama_sampler_sample(sampler, g_ctx, -1);
        if (new_tok == eos_token) break;

        int piece_len = llama_token_to_piece(
            vocab, new_tok,
            piece_buf, (int32_t)sizeof(piece_buf),
            /*lstrip=*/0, /*special=*/false
        );
        if (piece_len > 0) result.append(piece_buf, piece_len);

        llama_batch next = llama_batch_get_one(&new_tok, 1);
        if (llama_decode(g_ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);
    LOGI("Generated %zu chars", result.size());
    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_orag_ai_LlamaBridge_generateResponseStreaming(
        JNIEnv *env, jobject /*thiz*/, jstring jprompt, jobject jcallback) {
    if (!g_model || !g_ctx) return;

    const char *raw = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt(raw);
    env->ReleaseStringUTFChars(jprompt, raw);

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    std::vector<llama_token> tokens(2048);
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(), (int32_t)prompt.size(),
        tokens.data(), (int32_t)tokens.size(),
        /*add_special=*/true,
        /*parse_special=*/true
    );
    if (n_tokens < 0) return;
    tokens.resize(n_tokens);

    llama_kv_self_clear(g_ctx);
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(g_ctx, batch) != 0) return;

    llama_sampler *sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist((uint32_t)time(nullptr)));

    const llama_token eos_token = llama_vocab_eos(vocab);

    // Resolve the Kotlin StreamCallback.onToken method once
    jclass   cbClass  = env->GetObjectClass(jcallback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");

    char piece_buf[256];
    for (int i = 0; i < 256; ++i) {
        llama_token new_tok = llama_sampler_sample(sampler, g_ctx, -1);
        if (new_tok == eos_token) break;

        int piece_len = llama_token_to_piece(
            vocab, new_tok,
            piece_buf, (int32_t)(sizeof(piece_buf) - 1),
            /*lstrip=*/0, /*special=*/false
        );
        if (piece_len > 0) {
            piece_buf[piece_len] = '\0';
            jstring jpiece = env->NewStringUTF(piece_buf);
            env->CallVoidMethod(jcallback, onToken, jpiece);
            env->DeleteLocalRef(jpiece);
        }

        llama_batch next = llama_batch_get_one(&new_tok, 1);
        if (llama_decode(g_ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);
    LOGI("Streaming generation complete");
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_orag_ai_LlamaBridge_getEmbedding(JNIEnv *env, jobject /*thiz*/, jstring jtext) {
    const int n_embd = g_model ? llama_model_n_embd(g_model) : 768;

    if (!g_model || !g_ctx) {
        jfloatArray empty = env->NewFloatArray(n_embd);
        return empty;
    }

    const char *raw = env->GetStringUTFChars(jtext, nullptr);
    std::string text(raw);
    env->ReleaseStringUTFChars(jtext, raw);

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(
        vocab,
        text.c_str(), (int32_t)text.size(),
        tokens.data(), (int32_t)tokens.size(),
        true, false
    );
    if (n_tokens <= 0) {
        return env->NewFloatArray(n_embd);
    }
    tokens.resize(n_tokens);

    llama_kv_self_clear(g_ctx);
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    llama_decode(g_ctx, batch);

    // Use last-token embedding (compatible across llama.cpp versions)
    const float *embd = llama_get_embeddings_ith(g_ctx, n_tokens - 1);
    if (!embd) embd = llama_get_embeddings(g_ctx);

    jfloatArray result = env->NewFloatArray(n_embd);
    if (embd) {
        env->SetFloatArrayRegion(result, 0, n_embd, embd);
    }
    return result;
}
