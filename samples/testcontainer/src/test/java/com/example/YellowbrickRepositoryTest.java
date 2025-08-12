package com.example;

import com.yellowbrick.testcontainer.YellowbrickContainer;
import com.zaxxer.hikari.HikariDataSource;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.sql.DataSource;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for YellowbrickContainer using Spring Boot Test framework.
 *
 * <p>This test class demonstrates how to use the YellowbrickContainer for integration
 * testing with Spring Boot. It covers:</p>
 * <ul>
 *   <li>Container initialization and configuration</li>
 *   <li>Database connection and setup</li>
 *   <li>JDBC operations (CRUD)</li>
 *   <li>Yellowbrick-specific features and system tables</li>
 *   <li>Direct ybsql command execution</li>
 * </ul>
 *
 * <p>The test uses a non-web Spring Boot configuration to minimize startup time
 * and avoid unnecessary web dependencies.</p>
 *
 * @author Yellowbrick Spring AI Team
 */
@SpringBootTest(
        classes = YellowbrickRepositoryTest.TestConfig.class,
        webEnvironment = SpringBootTest.WebEnvironment.NONE,
        properties = {"spring.profiles.active=test"}
)
@Testcontainers
@Timeout(value = 15, unit = TimeUnit.MINUTES) // Yellowbrick requires extended startup time
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class YellowbrickRepositoryTest {

    /**
     * Minimal Spring Boot test configuration that provides only the beans
     * necessary for database testing. This configuration:
     * - Creates a HikariCP DataSource configured for the Yellowbrick container
     * - Provides a JdbcTemplate bean for database operations
     * - Excludes web-related auto-configuration to minimize startup time
     */
    @SpringBootConfiguration
    static class TestConfig {

        /**
         * Primary DataSource bean configured to connect to the Yellowbrick test container.
         * Uses HikariCP connection pooling with conservative timeout settings suitable
         * for test environments.
         *
         * @return configured DataSource pointing to the Yellowbrick container
         */
        @Bean
        @Primary
        public DataSource dataSource() {
            HikariDataSource dataSource = new HikariDataSource();
            dataSource.setJdbcUrl(yellowbrick.getJdbcUrl());
            dataSource.setUsername(yellowbrick.getUsername());
            dataSource.setPassword(yellowbrick.getPassword());
            dataSource.setDriverClassName("org.postgresql.Driver");

            // Conservative connection pool settings for test environment
            dataSource.setMaximumPoolSize(5);
            dataSource.setConnectionTimeout(60000); // 60 seconds - generous for container startup
            dataSource.setValidationTimeout(10000);  // 10 seconds - time to validate connections

            return dataSource;
        }

        /**
         * JdbcTemplate bean for executing SQL operations against the test database.
         *
         * @param dataSource the DataSource to use for database connections
         * @return configured JdbcTemplate instance
         */
        @Bean
        public JdbcTemplate jdbcTemplate(DataSource dataSource) {
            return new JdbcTemplate(dataSource);
        }
    }

    /**
     * Yellowbrick test container instance. This container:
     * - Starts a Yellowbrick Community Edition database
     * - Exposes ports for database (5432) and web interface (443) access
     * - Runs in privileged mode as required by Yellowbrick
     * - Logs all container output with [YELLOWBRICK] prefix for debugging
     *
     * The container is static and shared across all test methods for performance.
     */
    @Container
    static YellowbrickContainer yellowbrick = YellowbrickContainer.create()
            .withLogConsumer(outputFrame -> System.out.print("[YELLOWBRICK] " + outputFrame.getUtf8String()));

    /**
     * JdbcTemplate instance for executing database operations in tests.
     * Initialized in setUp() method before each test.
     */
    private JdbcTemplate jdbcTemplate;

    /**
     * Configure Spring application properties dynamically based on the running container.
     * This ensures that Spring's DataSource configuration matches the actual container
     * connection details (host, port, credentials).
     *
     * @param registry Spring's dynamic property registry for test configuration
     */
    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", yellowbrick::getJdbcUrl);
        registry.add("spring.datasource.username", yellowbrick::getUsername);
        registry.add("spring.datasource.password", yellowbrick::getPassword);
        registry.add("spring.datasource.driver-class-name", yellowbrick::getDriverClassName);
    }

    /**
     * Set up test environment before each test method.
     * This method performs the following initialization steps:
     * 1. Verify container is running and accessible
     * 2. Wait for Yellowbrick database to be fully ready
     * 3. Create and configure JdbcTemplate with retry logic
     * 4. Set up test data (tables and sample records)
     *
     * The setup is designed to be robust and handle the extended startup
     * time that Yellowbrick requires.
     */
    @BeforeEach
    void setUp() {
        logContainerStatus();
        verifyContainerIsRunning();
        waitForYellowbrickToBeReady();
        createJdbcTemplate();
        verifyDatabaseConnection();
        setupTestData();
    }

    /**
     * Log current container status for debugging purposes.
     * Provides visibility into container state and connection details.
     */
    private void logContainerStatus() {
        System.out.println("=== Container Status ===");
        System.out.println("Container running: " + yellowbrick.isRunning());
        System.out.println("Container JDBC URL: " + yellowbrick.getJdbcUrl());
        System.out.println("Web Interface URL: " + yellowbrick.getWebInterfaceUrl());
        System.out.println("========================");
    }

    /**
     * Verify that the container is running before proceeding with setup.
     *
     * @throws RuntimeException if container is not running
     */
    private void verifyContainerIsRunning() {
        if (!yellowbrick.isRunning()) {
            throw new RuntimeException("Yellowbrick container is not running!");
        }
    }

    /**
     * Wait for Yellowbrick database to be fully ready for connections.
     * This uses the container's built-in readiness check which verifies
     * that the cluster state is "RUNNING" via ybsql commands.
     */
    private void waitForYellowbrickToBeReady() {
        System.out.println("Waiting for Yellowbrick to be ready...");
        yellowbrick.waitUntilYellowbrickReady(Duration.ofMinutes(5));
        System.out.println("Yellowbrick is ready!");
    }

    /**
     * Create and configure JdbcTemplate instance for test use.
     * Uses HikariCP with conservative timeout settings appropriate
     * for the container environment.
     */
    private void createJdbcTemplate() {
        HikariDataSource dataSource = new HikariDataSource();
        dataSource.setJdbcUrl(yellowbrick.getJdbcUrl());
        dataSource.setUsername(yellowbrick.getUsername());
        dataSource.setPassword(yellowbrick.getPassword());
        dataSource.setDriverClassName("org.postgresql.Driver");

        // Test-appropriate connection pool settings
        dataSource.setMaximumPoolSize(5);
        dataSource.setConnectionTimeout(60000); // 60 seconds
        dataSource.setValidationTimeout(10000);  // 10 seconds

        jdbcTemplate = new JdbcTemplate(dataSource);
    }

    /**
     * Verify database connection with retry logic.
     * This provides resilience against temporary connection issues
     * that may occur during container startup.
     */
    private void verifyDatabaseConnection() {
        testConnectionWithRetry(jdbcTemplate, 5);
    }

    /**
     * Set up test data structure and sample records.
     * Creates a test table with distribution settings appropriate
     * for Yellowbrick and inserts sample data for test operations.
     */
    private void setupTestData() {
        System.out.println("Setting up test data...");

        // Drop existing table if present (clean slate for each test)
        jdbcTemplate.execute("DROP TABLE IF EXISTS test_users CASCADE");

        // Create test table with Yellowbrick distribution strategy
        jdbcTemplate.execute("""
            CREATE TABLE test_users (
                id INTEGER,
                name VARCHAR(255),
                email VARCHAR(255),
                age INTEGER
            ) DISTRIBUTE ON (id)
        """);

        // Insert sample test data
        jdbcTemplate.update(
                "INSERT INTO test_users (id, name, email, age) VALUES (?, ?, ?, ?)",
                1, "John Doe", "john@example.com", 30
        );
        jdbcTemplate.update(
                "INSERT INTO test_users (id, name, email, age) VALUES (?, ?, ?, ?)",
                2, "Jane Smith", "jane@example.com", 25
        );

        System.out.println("Test data setup complete.");
    }

    // === TEST METHODS ===

    /**
     * Test basic connectivity to the Yellowbrick database.
     * Verifies that we can establish a connection and execute
     * a simple query to retrieve the current database name.
     */
    @Test
    void shouldConnectToYellowbrick() {
        // Execute basic connectivity test
        String result = jdbcTemplate.queryForObject("SELECT current_database()", String.class);
        System.out.println("Connected to database: " + result);

        // Verify we got a valid database name
        assertThat(result).isNotNull();
    }

    /**
     * Test Yellowbrick-specific functionality by querying version information.
     * This verifies that we're actually connected to a Yellowbrick instance
     * rather than a generic PostgreSQL database.
     */
    @Test
    void shouldQueryYellowbrickVersion() {
        // Query version information to confirm Yellowbrick connectivity
        String version = jdbcTemplate.queryForObject("SELECT version()", String.class);
        System.out.println("Yellowbrick version: " + version);

        // Verify the version string indicates Yellowbrick
        assertThat(version).containsIgnoringCase("yellowbrick");
    }

    /**
     * Test basic SELECT operations on test data.
     * Verifies that we can query the test table and retrieve
     * the expected records with correct data types and values.
     */
    @Test
    void shouldFindAllUsers() {
        // Query all users ordered by name
        List<Map<String, Object>> users = jdbcTemplate.queryForList(
                "SELECT * FROM test_users ORDER BY name"
        );

        // Verify we have the expected number of records
        assertThat(users).hasSize(2);

        // Verify first user (Jane Smith, ordered by name)
        assertThat(users.get(0).get("name")).isEqualTo("Jane Smith");
        assertThat(users.get(0).get("email")).isEqualTo("jane@example.com");
        assertThat(users.get(0).get("age")).isEqualTo(25);
    }

    /**
     * Test INSERT operations and data persistence.
     * Verifies that we can insert new records and subsequently
     * retrieve them with the correct values.
     */
    @Test
    void shouldInsertNewUser() {
        // Insert new user record
        int rowsAffected = jdbcTemplate.update(
                "INSERT INTO test_users (id, name, email, age) VALUES (?, ?, ?, ?)",
                3, "Alice Johnson", "alice@example.com", 28
        );

        // Verify insert operation affected exactly one row
        assertThat(rowsAffected).isEqualTo(1);

        // Retrieve and verify the inserted record
        Map<String, Object> insertedUser = jdbcTemplate.queryForMap(
                "SELECT * FROM test_users WHERE id = ?", 3
        );
        assertThat(insertedUser.get("name")).isEqualTo("Alice Johnson");
    }

    /**
     * Test access to Yellowbrick system tables and metadata.
     * This verifies that we can query Yellowbrick-specific system
     * tables that provide database metadata and configuration information.
     */
    @Test
    void shouldExecuteYellowbrickSpecificQueries() {
        // Query Yellowbrick system tables for schema information
        List<Map<String, Object>> schemas = jdbcTemplate.queryForList(
                "SELECT name FROM sys.schema WHERE name NOT LIKE 'sys%'"
        );

        // Verify we can access system tables and get results
        assertThat(schemas).isNotEmpty();
        System.out.println("Available schemas: " + schemas);
    }

    /**
     * Test direct ybsql command execution within the container.
     * This verifies that we can execute database commands directly
     * using Yellowbrick's native ybsql client, which may be needed
     * for operations not available through JDBC.
     */
    @Test
    void shouldTestYellowbrickDistribution() throws Exception {
        // Execute query using ybsql command directly in container
        var result = yellowbrick.executeQuery(
                "SELECT COUNT(*) FROM test_users"
        );

        // Verify command executed successfully
        assertThat(result.getExitCode()).isEqualTo(0);
        System.out.println("ybsql output: " + result.getStdout());
    }

    // === UTILITY METHODS ===

    /**
     * Test database connectivity with retry logic to handle container startup delays.
     *
     * <p>This method implements a robust connection testing strategy that accounts
     * for the fact that Yellowbrick containers may take time to become fully ready
     * for connections even after the container reports as "started".</p>
     *
     * @param jdbcTemplate the JdbcTemplate to test
     * @param maxRetries maximum number of connection attempts
     * @throws RuntimeException if all connection attempts fail
     */
    private void testConnectionWithRetry(JdbcTemplate jdbcTemplate, int maxRetries) {
        System.out.println("Testing database connection with retry logic...");

        for (int i = 0; i < maxRetries; i++) {
            try {
                // Execute simple test query to verify connectivity
                jdbcTemplate.queryForObject("SELECT 1", Integer.class);
                System.out.println("Database connection successful on attempt " + (i + 1));
                return;

            } catch (Exception e) {
                System.out.println("Connection attempt " + (i + 1) + " failed: " + e.getMessage());

                // If this was the last attempt, fail the test
                if (i == maxRetries - 1) {
                    throw new RuntimeException("Failed to connect after " + maxRetries + " attempts", e);
                }

                // Wait before retrying (longer delay for Yellowbrick)
                try {
                    Thread.sleep(10000); // 10 seconds between attempts
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Interrupted during connection retry", ie);
                }
            }
        }
    }
}