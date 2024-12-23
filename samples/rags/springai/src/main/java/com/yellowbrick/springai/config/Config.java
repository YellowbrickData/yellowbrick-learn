package com.yellowbrick.springai.config;

import com.yellowbrick.springai.vectorstore.YellowBrickVectorStore;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

@Configuration
@EnableConfigurationProperties(YbVectorStoreProperties.class)
public class Config {

    @Bean
    @ConditionalOnMissingBean({BatchingStrategy.class})
    BatchingStrategy pgVectorStoreBatchingStrategy() {
        return new TokenCountBatchingStrategy();
    }

    @Bean
     ChatClient chatClient(ChatClient.Builder builder) {
        return builder.build();
    }
    @Bean
    YellowBrickVectorStore ybvectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel, YbVectorStoreProperties properties, ObjectProvider<ObservationRegistry> observationRegistry, ObjectProvider<VectorStoreObservationConvention> customObservationConvention, BatchingStrategy batchingStrategy) {
        return new YellowBrickVectorStore( properties.getTableName(), jdbcTemplate, embeddingModel, properties.isInitializeSchema(),
                (ObservationRegistry) observationRegistry.getIfUnique(() -> {
                    return ObservationRegistry.NOOP;
                }),
                (VectorStoreObservationConvention) customObservationConvention.getIfAvailable(() -> {
                    return null;
                }), batchingStrategy, properties.getMaxDocumentBatchSize());

    }
}
