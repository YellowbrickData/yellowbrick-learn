// DocumentRetrievalStrategy.java
package com.yellowbrick.springai.vectorstore;

import org.springframework.ai.document.Document;
import java.util.List;
import java.util.UUID;

/**
 * Strategy interface for different document retrieval implementations
 */
public interface DocumentRetrievalStrategy {

    /**
     * Retrieve documents based on the search criteria
     *
     * @param searchDocumentId Unique identifier for this search operation
     * @param embeddings Query embeddings to search against
     * @param filterString Additional filter criteria
     * @param topK Number of top results to return
     * @param vectorStore Reference to the vector store for accessing helper methods
     * @return List of matching documents with similarity scores
     */
    List<Document> getDocuments(UUID searchDocumentId, float[] embeddings, String filterString, int topK, YellowBrickVectorStore vectorStore);

    /**
     * Called after documents are added to perform any strategy-specific post-processing
     * (e.g., updating indexes, rebuilding caches, etc.)
     *
     * @param vectorStore Reference to the vector store
     * @param addedDocuments List of documents that were just added
     */
    default void onDocumentsAdded(YellowBrickVectorStore vectorStore, List<Document> addedDocuments) {
        // Default implementation does nothing
    }

    /**
     * Called after documents are deleted to perform any strategy-specific cleanup
     *
     * @param vectorStore Reference to the vector store
     * @param deletedIds List of document IDs that were deleted
     */
    default void onDocumentsDeleted(YellowBrickVectorStore vectorStore, List<String> deletedIds) {
        // Default implementation does nothing
    }
}