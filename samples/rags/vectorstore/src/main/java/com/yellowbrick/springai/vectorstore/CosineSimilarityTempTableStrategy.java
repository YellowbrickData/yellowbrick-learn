package com.yellowbrick.springai.vectorstore;

import org.springframework.stereotype.Component;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.StatementCreatorUtils;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Original implementation using cosine similarity with temporary tables
 */
@Component("cosineSimilarityTempTable")
public class CosineSimilarityTempTableStrategy implements DocumentRetrievalStrategy {

    private static final Logger logger = LoggerFactory.getLogger(CosineSimilarityTempTableStrategy.class);
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public List<Document> getDocuments(UUID searchDocumentId, float[] embeddings, String filterString, int topK, YellowBrickVectorStore vectorStore) {
        JdbcTemplate jdbcTemplate = vectorStore.getJdbcTemplate();

        // Create temporary table and insert search embeddings
        createTemporaryTable(searchDocumentId, embeddings, vectorStore);
        insertSearchDocEmbeddings(searchDocumentId, embeddings, vectorStore);

        // Execute the similarity search query
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
                "                " + vectorStore.getQueryTableName() +" v1 " +
                "                INNER JOIN" +
                "               " + vectorStore.getTableName() +" v2" +
                "                ON v1.embedding_id = v2.embedding_id" +
                "                where v1.doc_id = ?" +
                "                GROUP BY v2.doc_id" +
                "                ORDER BY score DESC LIMIT "+ topK +
                "        ) v4" +
                " INNER JOIN" +
                " " + vectorStore.getContentTableName()+" v3" +
                " ON v4.doc_id = v3.doc_id" +
                " WHERE 1=1" + filterString +
                " ORDER BY score DESC";

        return jdbcTemplate.query(selectSQL, new RowMapper<Document>() {
            @Override
            public Document mapRow(ResultSet rs, int rowNum) throws SQLException {
                Map<String, Object> result = null;
                try {
                    String res = rs.getString(2);
                    logger.debug(res);
                    result = objectMapper.readValue(rs.getString(2), Map.class);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
                return new Document(rs.getString(4), rs.getString(1), result);
            }
        }, new Object[]{searchDocumentId.toString()});
    }

    private void insertSearchDocEmbeddings(UUID searchDocumentId, float[] embeddings, YellowBrickVectorStore vectorStore) {
        String insertTemp = "INSERT INTO "+ vectorStore.getQueryTableName() + " (doc_id, embedding_id, embedding) VALUES (?,?,?)";
        vectorStore.getJdbcTemplate().batchUpdate(insertTemp, new BatchPreparedStatementSetter() {

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

    private void createTemporaryTable(UUID searchDocumentId, float[] embeddings, YellowBrickVectorStore vectorStore) {
        String tempTableCreate = String.format(
                " CREATE TEMPORARY TABLE  %s ( \n" +
                        "     doc_id UUID,\n" +
                        "     embedding_id SMALLINT,\n" +
                        "      embedding FLOAT)\n" +
                        "  ON COMMIT DROP\n"+
                        "  DISTRIBUTE REPLICATE\n", vectorStore.getQueryTableName());
        vectorStore.getJdbcTemplate().execute(tempTableCreate);
    }

}
