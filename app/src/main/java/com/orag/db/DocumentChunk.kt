package com.orag.db

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import java.util.UUID

@Entity(tableName = "document_chunks")
data class DocumentChunk(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val sourcePath: String,
    val chunkIndex: Int,
    val textContent: String,
    val embeddingVector: FloatArray // The Dense Vector from Nomic / Hybrid LLM
)

class FloatArrayConverter {
    @TypeConverter
    fun fromFloatArray(array: FloatArray?): String? {
        return array?.joinToString(separator = ",")
    }

    @TypeConverter
    fun toFloatArray(data: String?): FloatArray? {
        if (data.isNullOrEmpty()) return null
        return data.split(",").map { it.toFloat() }.toFloatArray()
    }
}
