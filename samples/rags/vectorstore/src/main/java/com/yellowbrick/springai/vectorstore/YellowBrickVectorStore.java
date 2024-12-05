package com.yellowbrick.springai.vectorstore;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;

import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.FilterExpressionConverter;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.StatementCreatorUtils;
import org.springframework.lang.Nullable;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionStatus;
import org.springframework.transaction.support.TransactionCallback;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.util.StringUtils;


import java.sql.*;
import java.util.*;

public class YellowBrickVectorStore extends AbstractObservationVectorStore implements InitializingBean {
    private static final Logger logger = LoggerFactory.getLogger(YellowBrickVectorStore.class);
    private final JdbcTemplate jdbcTemplate;
    private final BatchingStrategy batchingStrategy;
    private final String vectorTableName;
    private final EmbeddingModel embeddingModel;
    private final int maxDocumentBatchSize;
    private final boolean removeExistingVectorStoreTable;
    private final boolean initializeSchema;
    private final ObjectMapper objectMapper;
    private final TransactionTemplate transactionTemplate;
    public final FilterExpressionConverter filterExpressionConverter = new YbVectorFilterExpressionConverter();
    private Logger log = LoggerFactory.getLogger(YellowBrickVectorStore.class);

    public YellowBrickVectorStore(String vectorTableName, JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel, boolean initializeSchema, boolean dropSchema, ObservationRegistry observationRegistry, VectorStoreObservationConvention observationConvention, BatchingStrategy batchingStrategy, int maxDocumentBatchSize, PlatformTransactionManager transactionManager) {
        super(observationRegistry, observationConvention);
        this.jdbcTemplate = jdbcTemplate;
        this.embeddingModel = embeddingModel;
        this.batchingStrategy = batchingStrategy;
        this.maxDocumentBatchSize = maxDocumentBatchSize;
        this.vectorTableName = null != vectorTableName && !vectorTableName.isEmpty() ? vectorTableName.trim() : "vector_store";
        this.initializeSchema = initializeSchema;
        this.removeExistingVectorStoreTable = dropSchema;
        this.objectMapper = new ObjectMapper();
        this.transactionTemplate = new TransactionTemplate(transactionManager);


    }

    /**
     * Add a document to the vector store
     *
     * @param documents
     */
    @Override
    public void doAdd(List<Document> documents) {
        this.embeddingModel.embed(documents, EmbeddingOptionsBuilder.builder().build(), this.batchingStrategy);
        List<List<Document>> batchedDocuments = this.batchDocuments(documents);
        batchedDocuments.forEach(this::insertOrUpdateContent);
    }

    private String toJson(Map<String, Object> map) {
        try {
            return this.objectMapper.writeValueAsString(map);
        } catch (JsonProcessingException var3) {
            JsonProcessingException e = var3;
            throw new RuntimeException(e);
        }
    }

    private void insertOrUpdateEmbeddings(float[] embeddings, String doc_id) {
        String sql = "INSERT INTO " + getTableName() + "(doc_id, embedding_id, embedding) VALUES (?, ?,?)";

        this.jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {

            @Override
            public void setValues(PreparedStatement ps, int i) throws SQLException {
                StatementCreatorUtils.setParameterValue(ps, 1, Integer.MIN_VALUE, UUID.fromString(doc_id));
                StatementCreatorUtils.setParameterValue(ps, 2, Integer.MIN_VALUE, i);
                StatementCreatorUtils.setParameterValue(ps, 3, Integer.MIN_VALUE, embeddings[i]);

            }

            @Override
            public int getBatchSize() {
                return embeddings.length;
            }
        });

    }

    private void insertOrUpdateContent(List<Document> batch) {
        String sql = "INSERT INTO " + this.getContentTableName() + " (doc_id, text, metadata) VALUES (?, ?, ? ::jsonb )";

        this.jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {

            @Override
            public void setValues(PreparedStatement ps, int i) throws SQLException {

                Document document = batch.get(i);
                String content = document.getContent();
                String json = toJson(document.getMetadata());
                float[] embedding = document.getEmbedding();
                StatementCreatorUtils.setParameterValue(ps, 1, Integer.MIN_VALUE, UUID.fromString(document.getId()));
                StatementCreatorUtils.setParameterValue(ps, 2, Integer.MIN_VALUE, content);
                StatementCreatorUtils.setParameterValue(ps, 3, Integer.MIN_VALUE, json);
                insertOrUpdateEmbeddings(embedding, document.getId());

            }

            @Override
            public int getBatchSize() {
                return batch.size();
            }
        });
    }

    private List<List<Document>> batchDocuments(List<Document> documents) {
        List<List<Document>> batches = new ArrayList<>();

        for (int i = 0; i < documents.size(); i += this.maxDocumentBatchSize) {
            batches.add(documents.subList(i, Math.min(i + this.maxDocumentBatchSize, documents.size())));
        }

        return batches;
    }

    @Override
    public Optional<Boolean> doDelete(List<String> idList) {
        //AtomicInteger updateCount = new AtomicInteger();
        //increment atomic integer from inside lambda.
        long count =  idList.stream()
                .filter(id-> 1 == this.jdbcTemplate.update("DELETE FROM " + this.getTableName() + " WHERE id = ?", new Object[]{UUID.fromString(id)}))
                .count();

        logger.info("records deleted {}",count);
        //todo total deleted records and return true if total=sizeof(idList)
        return Optional.of(true);
    }

    @Override
    public List<Document> doSimilaritySearch(SearchRequest request) {

       int topK = request.getTopK();
        //create embeddings out of the search request

        String nativeFilterExpression = (request.getFilterExpression() != null)
                ? this.filterExpressionConverter.convertExpression(request.getFilterExpression()) : "";



        float[] embeddings = this.getQueryEmbedding(request.getQuery());
        UUID searchDocumentId = UUID.randomUUID();

        List<Document> query = (List<Document>) transactionTemplate.execute(new TransactionCallback<List<Document>>() {
            @Override
            public List<Document> doInTransaction(TransactionStatus status) {

                String jsonPathFilter = "";

                if (StringUtils.hasText(nativeFilterExpression)) {
                    jsonPathFilter =  "AND (" + nativeFilterExpression +")";
                }

                createTemporaryTable(searchDocumentId, embeddings);

                insertSearchDocEmbeddings(searchDocumentId, embeddings);

                List<Document> query = getDocuments(searchDocumentId,jsonPathFilter,topK);

                cleanUpTempTable(searchDocumentId);
                return query;
            }


        });
        return query;
    }

    private void insertSearchDocEmbeddings(UUID searchDocumentId, float[] embeddings) {
        String insertTemp = "INSERT INTO "+ getQueryTableName() + " (doc_id, embedding_id, embedding) VALUES (?,?,?)";
        jdbcTemplate.batchUpdate(insertTemp, new BatchPreparedStatementSetter() {

            @Override
            public void setValues(PreparedStatement ps, int i) throws SQLException {
                StatementCreatorUtils.setParameterValue(ps, 1, Integer.MIN_VALUE, searchDocumentId.toString());
                StatementCreatorUtils.setParameterValue(ps, 2, Integer.MIN_VALUE, i);
                StatementCreatorUtils.setParameterValue(ps, 3, Integer.MIN_VALUE, embeddings[i]);
            }

            @Override
            public int getBatchSize() {
                return embeddings.length;
            }
        });
    }

    private List<Document> getDocuments(UUID searchDocumentId, String filterString, int topK) {

        String selectSQL = " SELECT " +
                "        text," +
                "         metadata," +
                "        score," +
                "        v4.doc_id" +
                "  FROM" +
                "        (SELECT" +
                "                v2.doc_id doc_id," +
                "                SUM(v1.embedding * v2.embedding) /" +
                "                        (SQRT(SUM(v1.embedding * v1.embedding)) *" +
                "                                SQRT(SUM(v2.embedding * v2.embedding))) AS score" +
                "                FROM" +
                "                " + getQueryTableName() +" v1 " +
                "                INNER JOIN" +
                "               " + getTableName() +" v2" +
                "                ON v1.embedding_id = v2.embedding_id" +
                "                where v1.doc_id = ?" +
                "                GROUP BY v2.doc_id" +
                "                ORDER BY score DESC LIMIT "+ topK +
                "        ) v4" +
                " INNER JOIN" +
                " " + getContentTableName()+" v3" +
                " ON v4.doc_id = v3.doc_id" +
                " WHERE 1=1" + filterString +
                " ORDER BY score DESC";

                //1=1 to avoid situation where filter is empty string

        List<Document> query = jdbcTemplate.query(selectSQL, new RowMapper<Document>() {

            @Override
            public Document mapRow(ResultSet rs, int rowNum) throws SQLException {
                Map<String, Object> result =
                        null;
                try {
                    String res = rs.getString(2);
                    logger.debug(res);
                    result = new ObjectMapper().readValue(rs.getString(2), Map.class);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
                return new Document(rs.getString(4),rs.getString(1), result);
            }
        }, new Object[]{searchDocumentId.toString()});
        return query;
    }

    private void cleanUpTempTable(UUID searchDocumentId) {
        String deleteSQL = "DELETE FROM "+ getQueryTableName()+" where doc_id = ?";
        jdbcTemplate.update(deleteSQL, new Object[]{searchDocumentId.toString()});
    }

    private void createTemporaryTable(UUID searchDocumentId, float[] embeddings) {
        String tempTableCreate = String.format(
                " CREATE TEMPORARY TABLE  %s ( \n" +
                        "     doc_id UUID,\n" +
                        "     embedding_id SMALLINT,\n" +
                        "      embedding FLOAT)\n" +
                        "  ON COMMIT DROP\n"+
                        "  DISTRIBUTE REPLICATE\n", getQueryTableName());
        jdbcTemplate.execute(tempTableCreate);

    }

    private float[] getQueryEmbedding(String query) {
        return this.embeddingModel.embed(query);
    }

    @Override
    public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {
        //TODO add operationName to what is part of the observation context.  see https://github.com/spring-projects/spring-ai/issues/1204
        return VectorStoreObservationContext.builder(VectorStoreProvider.PG_VECTOR.value(), operationName)
                .withCollectionName(this.vectorTableName);


    }


    private String getVectorTableName() {
        return this.vectorTableName;
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        logger.info("Initializing PGVectorStore schema for table: {}", this.getVectorTableName());

        if (!this.initializeSchema) {
            logger.debug("Skipping the schema initialization for the table: {}", this.getTableName());
        } else {
            if (this.removeExistingVectorStoreTable) {
                this.jdbcTemplate.execute(String.format("DROP TABLE IF EXISTS %s", this.getTableName()));
                this.jdbcTemplate.execute(String.format("DROP TABLE IF EXISTS %s", this.getContentTableName()));

            }

            String c = getTableName() + "_pk_doc_id";

            this.jdbcTemplate.execute(String.format("  " +
                            "              CREATE TABLE IF NOT EXISTS %s (\n" +
                            "                doc_id UUID NOT NULL,\n" +
                            "                text VARCHAR(60000) NOT NULL,\n" +
                            "                metadata json NOT NULL,\n" +
                            "                CONSTRAINT %s PRIMARY KEY (doc_id))\n" +
                            "                DISTRIBUTE ON (doc_id) SORT ON (doc_id)"
                    , this.getContentTableName(), c));

            this.jdbcTemplate.execute(String.format("  " +
                            " CREATE TABLE IF NOT EXISTS %s (\n" +
                            " doc_id UUID NOT NULL,\n" +
                            " embedding_id SMALLINT NOT NULL,\n" +
                            " embedding FLOAT NOT NULL)\n"

                    , getTableName()));

        }
    }

    private String getTableName() {
        return this.vectorTableName;
    }


    private String getContentTableName() {
        return this.vectorTableName + "_content";
    }

    private String getQueryTableName() {
        return this.vectorTableName + "_query";
    }

    public static class Builder {
        private final JdbcTemplate jdbcTemplate;
        private final EmbeddingModel embeddingModel;
        private String schemaName = "public";
        private String vectorTableName;
        private boolean vectorTableValidationsEnabled = false;
        private int dimensions = -1;
        private boolean removeExistingVectorStoreTable;
        private boolean initializeSchema;
        private ObservationRegistry observationRegistry;
        private BatchingStrategy batchingStrategy;
        private int maxDocumentBatchSize;
        @Nullable
        private VectorStoreObservationConvention searchObservationConvention;
        private PlatformTransactionManager transactionManager = null;

        public Builder(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel) {
            this.removeExistingVectorStoreTable = false;
            this.observationRegistry = ObservationRegistry.NOOP;
            this.batchingStrategy = new TokenCountBatchingStrategy();
            this.maxDocumentBatchSize = 10000;
            this.jdbcTemplate = jdbcTemplate;
            this.embeddingModel = embeddingModel;

        }

        public Builder withSchemaName(String schemaName) {
            this.schemaName = schemaName;
            return this;
        }

        public Builder withVectorTableName(String vectorTableName) {
            this.vectorTableName = vectorTableName;
            return this;
        }

        public Builder withVectorTableValidationsEnabled(boolean vectorTableValidationsEnabled) {
            this.vectorTableValidationsEnabled = vectorTableValidationsEnabled;
            return this;
        }

        public Builder withDimensions(int dimensions) {
            this.dimensions = dimensions;
            return this;
        }


        public Builder withRemoveExistingVectorStoreTable(boolean removeExistingVectorStoreTable) {
            this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
            return this;
        }


        public Builder withInitializeSchema(boolean initializeSchema) {
            this.initializeSchema = initializeSchema;
            return this;
        }

        public Builder withObservationRegistry(ObservationRegistry observationRegistry) {
            this.observationRegistry = observationRegistry;
            return this;
        }

        public Builder withSearchObservationConvention(VectorStoreObservationConvention customObservationConvention) {
            this.searchObservationConvention = customObservationConvention;
            return this;
        }

        public Builder withBatchingStrategy(BatchingStrategy batchingStrategy) {
            this.batchingStrategy = batchingStrategy;
            return this;
        }

        public Builder withMaxDocumentBatchSize(int maxDocumentBatchSize) {
            this.maxDocumentBatchSize = maxDocumentBatchSize;
            return this;
        }

        public YellowBrickVectorStore build() {
            return new YellowBrickVectorStore( this.vectorTableName, this.jdbcTemplate, this.embeddingModel, this.initializeSchema,this.removeExistingVectorStoreTable, this.observationRegistry, this.searchObservationConvention, this.batchingStrategy, this.maxDocumentBatchSize, this.transactionManager);
        }

    }
}
