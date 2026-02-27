package com.orag.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(entities = [DocumentChunk::class], version = 1, exportSchema = false)
@TypeConverters(FloatArrayConverter::class)
abstract class AppDatabase : RoomDatabase() {

    abstract fun documentChunkDao(): DocumentChunkDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "orag_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
