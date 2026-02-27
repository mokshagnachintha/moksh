package com.orag.rag

import com.orag.ai.LlamaBridge
import com.orag.db.DocumentChunk
import com.orag.db.DocumentChunkDao
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/**
 * The core RAG logic that uses the ONE Hybrid AI model to do BOTH jobs
 * (Embedding and Generating) exactly as requested.
 */
class RagPipeline(
    private val llamaApi: LlamaBridge,
    private val chunkDao: DocumentChunkDao
) {

    /**
     * Translates a raw text document (like a PDF) into chunks and mathematical vectors.
     */
    suspend fun ingestDocument(sourceName: String, rawText: String) = withContext(Dispatchers.IO) {
        // Very basic chunking for example (every 200 words)
        val words = rawText.split(Regex("\\s+"))
        val chunks = words.chunked(200).map { it.joinToString(" ") }

        val dbChunks = chunks.mapIndexed { index, chunkText ->
            
            // MAGIC STEP #1: 
            // We use the Hybrid Model in "Encoder Mode" to get the embedding vector
            val trueVector = llamaApi.getEmbedding(chunkText)

            DocumentChunk(
                sourcePath = sourceName,
                chunkIndex = index,
                textContent = chunkText,
                embeddingVector = trueVector // Saving the brilliant vector, NOT TF-IDF!
            )
        }

        // Save into the Android SQLite Room Database
        chunkDao.insertAll(dbChunks)
    }

    /**
     * Answers a user question by searching the database and generating a response.
     */
    suspend fun answerQuestion(userQuery: String): String = withContext(Dispatchers.IO) {
        
        // MAGIC STEP #2:
        // Use the EXACT SAME Hybrid Model again to embed the user's question
        val queryVector = llamaApi.getEmbedding(userQuery)

        // Fetch all document chunks from the database
        val allChunks = chunkDao.getAllChunks()
        if (allChunks.isEmpty()) {
            return@withContext "I have no documents loaded to answer this question."
        }

        // Perform pure mathematical Vector Similarity Search (Cosine Similarity)
        // Find the 3 most relevant chunks to the question
        val topChunks = allChunks.map { chunk ->
            Pair(chunk, cosineSimilarity(queryVector, chunk.embeddingVector))
        }
        .sortedByDescending { it.second } // Sort by highest math score
        .take(3)
        .map { it.first }

        // Construct the RAG Prompt for the LLM
        val contextText = topChunks.joinToString("\n\n") { "Excerpt:\n${it.textContent}" }
        val finalPrompt = """
            You are a brilliant, concise AI assistant running offline.
            Use the following context to answer the user's question. If the answer is not in the context, do not make it up.
            
            === Context ===
            $contextText
            
            === User Question ===
            $userQuery
            
            Answer:
        """.trimIndent()

        // MAGIC STEP #3:
        // Use the EXACT SAME Hybrid Model in "Decoder Mode" (Normal LLM) 
        // to read the prompt and type out the final answer for the user!
        return@withContext llamaApi.generateResponse(finalPrompt)
    }

    /**
     * Standard Cosine Similarity formula to compare how statistically identical two float vectors are.
     * Returns 1.0 if they are identical, 0.0 if they are completely unrelated.
     */
    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in v1.indices) {
            dotProduct += v1[i] * v2[i]
            normA += v1[i] * v1[i]
            normB += v2[i] * v2[i]
        }
        return if (normA == 0f || normB == 0f) 0f else (dotProduct / (sqrt(normA) * sqrt(normB)))
    }
}
