package com.yellowbrick.springai.config;

import com.yellowbrick.springai.vectorstore.YellowBrickVectorStore;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.jdbc.JdbcTemplateAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.PlatformTransactionManager;

import javax.sql.DataSource;


@AutoConfiguration(after = JdbcTemplateAutoConfiguration.class)
@ConditionalOnClass({ YellowBrickVectorStore.class, DataSource.class, JdbcTemplate.class })
@EnableConfigurationProperties(YbVectorStoreProperties.class)
public class YbVectorStoreAutoConfig {

    @Bean
    @ConditionalOnMissingBean({BatchingStrategy.class})
    BatchingStrategy pgVectorStoreBatchingStrategy() {
        return new TokenCountBatchingStrategy();
    }

    @Bean
    @ConditionalOnMissingBean
    YellowBrickVectorStore ybvectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel, YbVectorStoreProperties properties, ObjectProvider<ObservationRegistry> observationRegistry, ObjectProvider<VectorStoreObservationConvention> customObservationConvention, BatchingStrategy batchingStrategy, PlatformTransactionManager platformTransactionManager) {
        return new YellowBrickVectorStore( properties.getTableName(), jdbcTemplate, embeddingModel, properties.isInitializeSchema(), properties.isRemoveExistingVectorStoreTable(),
                (ObservationRegistry) observationRegistry.getIfUnique(() -> {
                    return ObservationRegistry.NOOP;
                }),
                (VectorStoreObservationConvention) customObservationConvention.getIfAvailable(() -> {
                    return null;
                }), batchingStrategy, properties.getMaxDocumentBatchSize(),platformTransactionManager);

    }
}
