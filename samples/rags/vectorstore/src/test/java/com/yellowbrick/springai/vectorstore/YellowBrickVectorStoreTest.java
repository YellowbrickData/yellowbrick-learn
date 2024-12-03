package com.yellowbrick.springai.vectorstore;

import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.mockito.ArgumentCaptor;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;


public class YellowBrickVectorStoreTest {

    @Test
    void shouldAddDocumentsInBatchesAndEmbedOnce() {
        var jdbcTemplate = mock(JdbcTemplate.class);
        var embeddingModel = mock(EmbeddingModel.class);
        var documents = Collections.nCopies(9989, new Document("foo"));


        var YellowBrickVectorStore = new YellowBrickVectorStore.Builder(jdbcTemplate, embeddingModel).withMaxDocumentBatchSize(1000)
                .build();
        YellowBrickVectorStore.doAdd(documents);


        verify(embeddingModel, only()).embed(eq(documents), any(), any());


        var batchUpdateCaptor = ArgumentCaptor.forClass(BatchPreparedStatementSetter.class);
        verify(jdbcTemplate, times(10)).batchUpdate(anyString(), batchUpdateCaptor.capture());


        assertThat(batchUpdateCaptor.getAllValues()).hasSize(10)
                .allSatisfy(BatchPreparedStatementSetter::getBatchSize)
                .satisfies(batches -> {
                    for (int i = 0; i < 9; i++) {
                        assertThat(batches.get(i).getBatchSize()).as("Batch at index %d should have size 10", i)
                                .isEqualTo(1000);
                    }
                    assertThat(batches.get(9).getBatchSize()).as("Last batch should have size 989").isEqualTo(989);
                });

    }
}