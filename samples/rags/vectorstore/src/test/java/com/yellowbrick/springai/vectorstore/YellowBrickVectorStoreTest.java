package com.yellowbrick.springai.vectorstore;

import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.mockito.ArgumentCaptor;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;


public class YellowBrickVectorStoreTest {
    @Test
    void initShouldInitializeDropTables() throws Exception {
        var jdbcTemplate = mock(JdbcTemplate.class);
        var embeddingModel = mock(EmbeddingModel.class);
        var YellowBrickVectorStore = new YellowBrickVectorStore.Builder(jdbcTemplate, embeddingModel).withRemoveExistingVectorStoreTable(true).withInitializeSchema(true).withMaxDocumentBatchSize(1000)
                .build();
        YellowBrickVectorStore.afterPropertiesSet();
        verify(jdbcTemplate, times(1)).execute("DROP TABLE IF EXISTS vector_store");
        verify(jdbcTemplate,times(1)).execute("DROP TABLE IF EXISTS vector_store_content");
        verify(jdbcTemplate,times(1)).execute(contains("embedding FLOAT NOT NULL"));
        verify(jdbcTemplate,times(1)).execute(contains("DISTRIBUTE ON (doc_id) SORT ON (doc_id)"));

    }

    @Test
    void initShouldInitializeNotDropTables() throws Exception {
        var jdbcTemplate = mock(JdbcTemplate.class);
        var embeddingModel = mock(EmbeddingModel.class);
        var YellowBrickVectorStore = new YellowBrickVectorStore.Builder(jdbcTemplate, embeddingModel).withRemoveExistingVectorStoreTable(false).withInitializeSchema(true).withMaxDocumentBatchSize(1000)
                .build();
        YellowBrickVectorStore.afterPropertiesSet();
        verify(jdbcTemplate, never()).execute("DROP TABLE IF EXISTS vector_store");
        verify(jdbcTemplate,never()).execute("DROP TABLE IF EXISTS vector_store_content");
        verify(jdbcTemplate,times(1)).execute(contains("embedding FLOAT NOT NULL"));
        verify(jdbcTemplate,times(1)).execute(contains("DISTRIBUTE ON (doc_id) SORT ON (doc_id)"));

    }

    @Test
    void shouldInsertDocument() throws SQLException {
        var jdbcTemplate = mock(JdbcTemplate.class);
        var embeddingModel = mock(EmbeddingModel.class);
        var document = new Document("foo");
        float f[] = { 1.0f, 2.0f, 3.0f };
        document.setEmbedding(f);
        var documents = Collections.nCopies(1, document);
        doAnswer(invocationOnMock -> {
            PreparedStatement preparedStatementMock=mock(PreparedStatement.class);
            BatchPreparedStatementSetter setter = invocationOnMock.getArgument(1);
            setter.setValues(preparedStatementMock, 0);

            verify(preparedStatementMock,atLeast(0)).setObject(anyInt(),any(UUID.class));
            verify(preparedStatementMock,atLeast(0)).setString(anyInt(),any(String.class));
            verify(preparedStatementMock,atLeast(0)).setFloat(anyInt(),any(Float.class));

            return null;
                }).when(jdbcTemplate).batchUpdate(anyString(),any(BatchPreparedStatementSetter.class));


        var YellowBrickVectorStore = new YellowBrickVectorStore.Builder(jdbcTemplate, embeddingModel).withMaxDocumentBatchSize(1000)
                .build();
        YellowBrickVectorStore.doAdd(documents);

    }
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