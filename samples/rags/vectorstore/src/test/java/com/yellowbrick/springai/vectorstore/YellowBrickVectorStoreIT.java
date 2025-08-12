package com.yellowbrick.springai.vectorstore;

import com.yellowbrick.testcontainer.YellowbrickContainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.util.StreamUtils;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for YellowBrick Vector Store functionality with Spring AI.
 *
 * <p>This test class validates the complete integration between:</p>
 * <ul>
 *   <li>YellowbrickContainer (Testcontainers-based Yellowbrick database)</li>
 *   <li>YellowBrickVectorStore (Spring AI vector store implementation)</li>
 *   <li>OpenAI embedding model for document vectorization</li>
 *   <li>Vector similarity search and filtering capabilities</li>
 * </ul>
 *
 * <p>The test requires an OpenAI API key to be set in the OPENAI_API_KEY environment
 * variable, as it uses OpenAI's embedding model to generate vectors for documents.</p>
 *
 * <p>Test scenarios covered:</p>
 * <ul>
 *   <li>Document storage and vector similarity search</li>
 *   <li>Metadata-based filtering during search operations</li>
 *   <li>Schema initialization and table management</li>
 *   <li>Integration with Spring Boot auto-configuration</li>
 * </ul>
 *
 * @author Yellowbrick Spring AI Team
 */
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
@Testcontainers
public class YellowBrickVectorStoreIT {

    /**
     * Yellowbrick database container for testing.
     *
     * <p>Configured with:</p>
     * <ul>
     *   <li>Custom credentials for test isolation</li>
     *   <li>Bootstrap SQL script for initial schema setup</li>
     *   <li>Container logging for debugging</li>
     * </ul>
     *
     * <p>The container automatically starts before tests and stops after completion.</p>
     */
    @Container
    static YellowbrickContainer yellowbrick = YellowbrickContainer.create()
            .withPassword("mypass")
            .withUsername("myuser")
            .withDatabaseName("database_name")
            .withBootstrapData("yellowbrick/sql/init.sql")
            .withLogConsumer(outputFrame -> System.out.print("[YELLOWBRICK] " + outputFrame.getUtf8String()));

    /**
     * Spring Boot test context runner for isolated application context testing.
     *
     * <p>Configures:</p>
     * <ul>
     *   <li>Uses Spring Boot's DataSourceAutoConfiguration with dynamic properties</li>
     *   <li>Uses OpenAI auto-configuration with API key from properties</li>
     *   <li>Vector store schema initialization</li>
     *   <li>Automatic cleanup of existing vector tables</li>
     * </ul>
     */
    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(TestApplication.class)
            .withPropertyValues(
                    // Database connection configuration for test container
                    String.format("spring.datasource.url=%s", yellowbrick.getJdbcUrl()),
                    String.format("spring.datasource.username=%s", yellowbrick.getUsername()),
                    String.format("spring.datasource.password=%s", yellowbrick.getPassword()),


                    // OpenAI configuration for auto-configuration
                    String.format("spring.ai.openai.api-key=%s", System.getenv("OPENAI_API_KEY")),

                    // Vector store configuration for testing
                    "spring.ai.vectorstore.ybvector.initialize-schema=true",
                    "spring.ai.vectorstore.ybvector.remove-existing-vector-store-table=true"
            );

    /**
     * Test documents with various content and metadata for vector operations.
     *
     * <p>Each document includes:</p>
     * <ul>
     *   <li>Unique UUID identifier</li>
     *   <li>Text content loaded from classpath resources</li>
     *   <li>Metadata for testing filtering capabilities</li>
     * </ul>
     *
     * <p>The documents cover different topics to test similarity search accuracy.</p>
     */
    private final List<Document> documents = List.of(
            new Document(
                    UUID.randomUUID().toString(),
                    getText("classpath:/test/data/spring.ai.txt"),
                    Map.of("file_name", "spring.ai.txt", "meta", "meta1")
            ),
            new Document(
                    UUID.randomUUID().toString(),
                    getText("classpath:/test/data/time.shelter.txt"),
                    Map.of("file_name", "time.shelter.txt", "meta", "none")
            ),
            new Document(
                    UUID.randomUUID().toString(),
                    getText("classpath:/test/data/great.depression.txt"),
                    Map.of("file_name", "great.depression.txt", "meta", "meta2")
            ),
            new Document(
                    UUID.randomUUID().toString(),
                    getText("classpath:/test/data/great.depression.txt"),
                    Map.of("file_name", "great.depression2.txt", "meta", "meta3")
            )
    );

    /**
     * Test basic document storage and similarity search functionality.
     *
     * <p>This test verifies:</p>
     * <ul>
     *   <li>Documents can be successfully added to the vector store</li>
     *   <li>Similarity search returns relevant results</li>
     *   <li>Search results include expected content and metadata</li>
     *   <li>Top-K limiting works correctly</li>
     * </ul>
     *
     * <p>Expected behavior: Searching for "Great Depression" should return
     * the document about the Great Depression with the highest similarity score.</p>
     */
    @Test
    public void addAndSearchTest() {
        this.contextRunner.run(context -> {
            // Get the vector store bean from the application context
            VectorStore vectorStore = context.getBean(VectorStore.class);

            // Add all test documents to the vector store
            vectorStore.add(documents);

            // Perform similarity search for content related to "Great Depression"
            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.query("Great Depression").withTopK(1)
            );

            // Verify search returns exactly one result
            assertThat(results).hasSize(1);

            // Verify the returned document contains expected content
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getContent()).contains("The Great Depression (1929–1939) was an economic shock");

            // Verify metadata is preserved (original metadata + distance score)
            assertThat(resultDoc.getMetadata()).hasSize(2);
            assertThat(resultDoc.getMetadata()).containsKey("meta");
        });
    }

    /**
     * Test similarity search with metadata filtering capabilities.
     *
     * <p>This test verifies:</p>
     * <ul>
     *   <li>Filter expressions can be applied to search operations</li>
     *   <li>Only documents matching the filter criteria are returned</li>
     *   <li>Filtering works correctly with similarity search</li>
     *   <li>Specific document identification through metadata filtering</li>
     * </ul>
     *
     * <p>Expected behavior: Searching for "Great Depression" with a filter for
     * meta="meta2" should return only the specific document with that metadata value.</p>
     */
    @Test
    public void addAndSearchTestWithFilter() {
        this.contextRunner.run(context -> {
            // Get the vector store bean from the application context
            VectorStore vectorStore = context.getBean(VectorStore.class);

            // Build filter expression to match documents with meta="meta2"
            var filterBuilder = new FilterExpressionBuilder();
            var filterExpression = filterBuilder.eq("meta", "meta2").build();

            // Add all test documents to the vector store
            vectorStore.add(documents);

            // Perform similarity search with metadata filter applied
            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.query("Great Depression")
                            .withFilterExpression(filterExpression)
                            .withTopK(2) // Request up to 2 results, but filter should limit to 1
            );

            // Verify filter correctly limits results to one document
            assertThat(results).hasSize(1);

            // Verify the specific document with meta="meta2" was returned
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
            assertThat(resultDoc.getContent()).contains("The Great Depression (1929–1939) was an economic shock");

            // Verify metadata is preserved and includes the filter criteria
            assertThat(resultDoc.getMetadata()).hasSize(2);
            assertThat(resultDoc.getMetadata()).containsKey("meta");
        });
    }

    /**
     * Fully auto-configured test application that leverages all Spring Boot auto-configurations.
     *
     * <p>This configuration:</p>
     * <ul>
     *   <li>Uses Spring Boot's DataSourceAutoConfiguration for DataSource and JdbcTemplate</li>
     *   <li>Uses Spring Boot's OpenAiAutoConfiguration for EmbeddingModel</li>
     *   <li>Provides only custom beans specific to YellowBrick vector store</li>
     * </ul>
     */
    @SpringBootConfiguration
    @EnableAutoConfiguration
    public static class TestApplication {


    }

    /**
     * Utility method to load text content from classpath resources.
     *
     * <p>This method is used to load test documents from resource files,
     * allowing the test to work with realistic document content without
     * embedding large text strings directly in the test code.</p>
     *
     * @param uri the resource URI (e.g., "classpath:/test/data/document.txt")
     * @return the text content of the resource
     * @throws RuntimeException if the resource cannot be loaded
     */
    public static String getText(String uri) {
        var resource = new DefaultResourceLoader().getResource(uri);
        try {
            return resource.getContentAsString(StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load resource: " + uri, e);
        }
    }
}