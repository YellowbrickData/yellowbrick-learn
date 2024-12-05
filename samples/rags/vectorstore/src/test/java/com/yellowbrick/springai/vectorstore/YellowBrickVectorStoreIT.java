package com.yellowbrick.springai.vectorstore;

import com.yellowbrick.springai.config.YbVectorStoreAutoConfig;
import com.yellowbrick.springai.config.YbVectorStoreProperties;
import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
//import org.springframework.ai.document.DocumentMetadata;

import org.springframework.ai.autoconfigure.openai.OpenAiAutoConfiguration;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.PlatformTransactionManager;
import javax.sql.DataSource;
import com.zaxxer.hikari.HikariDataSource;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
public class YellowBrickVectorStoreIT {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(TestApplication.class)
            .withPropertyValues(// JdbcTemplate configuration
                    String.format("app.datasource.url=jdbc:postgresql://%s:%d/%s", System.getenv("YBHOST"),
                            5432, System.getenv("YBDATABASE")),
                    String.format("app.datasource.username=%s", System.getenv("YBUSER")),
                    String.format("app.datasource.password=%s",System.getenv("YBPASSWORD")),
                    "app.datasource.type=com.zaxxer.hikari.HikariDataSource",
                           "spring.ai.vectorstore.ybvector.initialize-schema=true",
                    "spring.ai.vectorstore.ybvector.remove-existing-vector-store-table=true");

    List<Document> documents = List.of(
            new Document(UUID.randomUUID().toString(), getText("classpath:/test/data/spring.ai.txt"), Map.of("meta1", "meta1")),
            new Document(UUID.randomUUID().toString(), getText("classpath:/test/data/time.shelter.txt"), Map.of()),
            new Document(UUID.randomUUID().toString(), getText("classpath:/test/data/great.depression.txt"), Map.of("meta2", "meta2")));



    public static String getText(String uri) {
        var resource = new DefaultResourceLoader().getResource(uri);
        try {
            return resource.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void addAndSearchTest() {
        this.contextRunner.run(context -> {
            this.documents.forEach(System.out::println);
            VectorStore vectorStore = context.getBean(VectorStore.class);
            vectorStore.add(documents);

            List<Document> results = vectorStore.similaritySearch(SearchRequest.query("Great Depression").withTopK(1));
            assertThat(results).hasSize(1);
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
            assertThat(resultDoc.getContent()).contains("The Great Depression (1929–1939) was an economic shock");
            assertThat(resultDoc.getMetadata()).hasSize(1);
            assertThat(resultDoc.getMetadata()).containsKey("meta2");

        });
    }


    @SpringBootConfiguration
    @EnableAutoConfiguration(exclude = { DataSourceAutoConfiguration.class, OpenAiAutoConfiguration.class })
    public static class TestApplication {

        @Bean
        public JdbcTemplate myJdbcTemplate(DataSource dataSource) {
            return new JdbcTemplate(dataSource);
        }

        @Bean
        public HikariDataSource dataSource(DataSourceProperties dataSourceProperties) {
            return dataSourceProperties.initializeDataSourceBuilder().type(HikariDataSource.class).build();
        }

        @Bean
        @Primary
        @ConfigurationProperties("app.datasource")
        public DataSourceProperties dataSourceProperties() {
            return new DataSourceProperties();
        }

        @Bean
        public EmbeddingModel embeddingModel() {
            return new OpenAiEmbeddingModel(new OpenAiApi(System.getenv("OPENAI_API_KEY")));
        }

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
}


