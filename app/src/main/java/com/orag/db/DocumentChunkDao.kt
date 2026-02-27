package com.orag.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface DocumentChunkDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertChunk(chunk: DocumentChunk)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(chunks: List<DocumentChunk>)

    @Query("SELECT * FROM document_chunks")
    suspend fun getAllChunks(): List<DocumentChunk>

    @Query("DELETE FROM document_chunks")
    suspend fun deleteAll()
}
