package com.orag

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateListOf
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val messages = remember { mutableStateListOf<Message>() }
                    var isLoading = remember { androidx.compose.runtime.mutableStateOf(true) }

                    // We run the heavy 1.1GB file copy on a background thread
                    androidx.compose.runtime.LaunchedEffect(Unit) {
                        withContext(Dispatchers.IO) {
                            val modelName = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
                            val outFile = java.io.File(filesDir, modelName)
                            
                            // On the very first launch, copy the 1.1GB model out of the hidden APK zip into pure storage
                            if (!outFile.exists()) {
                                assets.open(modelName).use { inputStream ->
                                    java.io.FileOutputStream(outFile).use { outputStream ->
                                        inputStream.copyTo(outputStream)
                                    }
                                }
                            }
                            
                            // Load the true physical file path into the C++ Engine
                            llamaApi.loadModel(outFile.absolutePath)
                        }
                        isLoading.value = false
                    }

                    ChatScreen(
                        messages = messages,
                        isModelLoading = isLoading.value,
                        onSendMessage = { userText ->
                            messages.add(Message("user", userText))
                            isLoading.value = true
                            
                            // Offload the heavy AI math to a background C++ thread so the UI stays 60fps
                            CoroutineScope(Dispatchers.IO).launch {
                                // 1. Calculate Vector Embedding for user text
                                val questionVector = llamaApi.getEmbedding(userText)
                                
                                // (If we had a real database hooked up, we would cosine similarity search here)
                                
                                // 2. Generate the Answer
                                val answer = llamaApi.generateResponse("Answer this: $userText")
                                
                                // 3. Push back to the main UI Screen
                                withContext(Dispatchers.Main) {
                                    messages.add(Message("assistant", answer))
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
