package com.orag

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import com.orag.ui.ChatScreen
import com.orag.ui.Message
import com.orag.ai.LlamaBridge
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private val llamaApi = LlamaBridge()

    /** Build the Llama-3.2 multi-turn chat template from full conversation history.
     *  Format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
     *  BOS (<|begin_of_text|>) is added automatically by llama_tokenize(add_special=true)
     */
    private fun buildPrompt(history: List<Message>): String {
        val sb = StringBuilder()
        sb.append("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n")
        val recentHistory = if (history.size > 12) history.takeLast(12) else history
        for (msg in recentHistory) {
            sb.append("<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id|>\n")
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return sb.toString()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val messages  = remember { mutableStateListOf<Message>() }
                    val isLoading = remember { mutableStateOf(true) }
                    val statusMsg = remember { mutableStateOf("Initialising model…") }

                    androidx.compose.runtime.LaunchedEffect(Unit) {
                        withContext(Dispatchers.IO) {
                            // Llama-3.2-1B Q4_K_M: 1B params (33% fewer than current 1.5B)
                            // = faster tok/s, same ARM dotprod kernel, Q4_K_M optimal dequant
                            // Public GGUF, no HF login required
                            val modelName = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                            val outFile   = java.io.File(filesDir, modelName)

                            // Download model on first launch
                            if (!outFile.exists()) {
                                withContext(Dispatchers.Main) {
                                    statusMsg.value = "Downloading model (≈770 MB)…"
                                }
                                val url = java.net.URL(
                                    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF" +
                                    "/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                                )
                                url.openStream().use { input ->
                                    java.io.FileOutputStream(outFile).use { output ->
                                        input.copyTo(output)
                                    }
                                }
                            }

                            withContext(Dispatchers.Main) {
                                statusMsg.value = "Loading model into memory…"
                            }
                            val ok = llamaApi.loadModel(outFile.absolutePath)

                            withContext(Dispatchers.Main) {
                                if (ok) {
                                    statusMsg.value = "Model ready"
                                    isLoading.value = false
                                } else {
                                    statusMsg.value = "Failed to load model"
                                    // keep isLoading = true so Send stays disabled
                                }
                            }
                        }
                    }

                    ChatScreen(
                        messages      = messages,
                        isModelLoading = isLoading.value,
                        statusMessage  = statusMsg.value,
                        onSendMessage  = { userText ->
                            messages.add(Message("user", userText))
                            isLoading.value = true
                            // Snapshot the full history (including the new user message)
                            // so the model sees the complete conversation context
                            val historySnapshot = messages.toList()

                            CoroutineScope(Dispatchers.IO).launch {
                                val prompt = buildPrompt(historySnapshot)

                                // Add an empty assistant bubble immediately so the user
                                // sees text appear token-by-token instead of waiting
                                withContext(Dispatchers.Main) {
                                    messages.add(Message("assistant", ""))
                                }

                                val sb = StringBuilder()
                                llamaApi.generateResponseStreaming(prompt) { piece ->
                                    sb.append(piece)
                                    val current = sb.toString()
                                    // Update the live assistant bubble on the UI thread
                                    this@MainActivity.runOnUiThread {
                                        val lastIdx = messages.lastIndex
                                        if (lastIdx >= 0 && messages[lastIdx].role == "assistant") {
                                            messages[lastIdx] = Message("assistant", current)
                                        }
                                    }
                                }

                                withContext(Dispatchers.Main) {
                                    // Trim trailing whitespace on the completed message
                                    val lastIdx = messages.lastIndex
                                    if (lastIdx >= 0 && messages[lastIdx].role == "assistant") {
                                        messages[lastIdx] = Message("assistant", messages[lastIdx].content.trim())
                                    }
                                    isLoading.value = false
                                }
                            }
                        }
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        llamaApi.unloadModel()
    }
}
