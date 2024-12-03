package com.yellowbrick.springai.config;

import org.springframework.ai.autoconfigure.vectorstore.CommonVectorStoreProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("spring.ai.vectorstore.ybvector")
public class YbVectorStoreProperties extends CommonVectorStoreProperties {
    public static final String CONFIG_PREFIX = "spring.ai.vectorstore.ybvector";
    private boolean removeExistingVectorStoreTable;
    private String tableName;
    private String schemaName;
    private int maxDocumentBatchSize;


    public YbVectorStoreProperties() {
        this.removeExistingVectorStoreTable = false;
        this.tableName = "vector_store";
        this.schemaName = "public";
        this.maxDocumentBatchSize = 10000;
    }







    public boolean isRemoveExistingVectorStoreTable() {
        return this.removeExistingVectorStoreTable;
    }

    public void setRemoveExistingVectorStoreTable(boolean removeExistingVectorStoreTable) {
        this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
    }

    public String getTableName() {
        return this.tableName;
    }

    public void setTableName(String vectorTableName) {
        this.tableName = vectorTableName;
    }

    public String getSchemaName() {
        return this.schemaName;
    }

    public void setSchemaName(String schemaName) {
        this.schemaName = schemaName;
    }



    public int getMaxDocumentBatchSize() {
        return this.maxDocumentBatchSize;
    }

    public void setMaxDocumentBatchSize(int maxDocumentBatchSize) {
        this.maxDocumentBatchSize = maxDocumentBatchSize;
    }
}
