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

    /** Build the Qwen2.5-Instruct multi-turn chat template from full conversation history. */
    private fun buildQwenPrompt(history: List<Message>): String {
        val sb = StringBuilder()
        sb.append("<|im_start|>system\nYou are a helpful assistant. Answer the user's questions directly and concisely.<|im_end|>\n")
        for (msg in history) {
            sb.append("<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
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
                            val modelName = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
                            val outFile   = java.io.File(filesDir, modelName)

                            // Download model on first launch
                            if (!outFile.exists()) {
                                withContext(Dispatchers.Main) {
                                    statusMsg.value = "Downloading model (≈900 MB)…"
                                }
                                val url = java.net.URL(
                                    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF" +
                                    "/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
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
                                val prompt = buildQwenPrompt(historySnapshot)
                                val answer = llamaApi.generateResponse(prompt)

                                withContext(Dispatchers.Main) {
                                    messages.add(Message("assistant", answer.trim()))
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
